from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import polars as pl
from gensim.models.keyedvectors import KeyedVectors
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

from readnext.modeling.language_models.bm25 import bm25
from readnext.utils.aliases import Embedding, EmbeddingsFrame, Tokens, TokensFrame
from readnext.utils.progress_bar import rich_progress_bar


class AggregationStrategy(str, Enum):
    """
    Determines the aggregation strategy for computing document embeddings from token embeddings.

    - `mean` sets the average of all token embeddings in a document as the document embedding.
    - `max` sets the maximum of all token embeddings in a document as the document embedding.
    """

    mean = "mean"
    max = "max"  # noqa: A003

    def __str__(self) -> str:
        return self.value

    @property
    def is_mean(self) -> bool:
        return self == self.mean

    @property
    def is_max(self) -> bool:
        return self == self.max


@dataclass(kw_only=True)
class Embedder(ABC):
    """
    Abstract Base class for embedding models. All embedding models implement
    `compute_embedding_single_document()` and `compute_embeddings_frame()` methods.
    """

    tokens_frame: TokensFrame

    @abstractmethod
    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        """
        Computes the embedding (list of floats) for a single document.
        """

    @abstractmethod
    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        """
        Takes a tokens frame with the two columns `d3_document_id` and `tokens` as input
        and computes the abstract embeddings for all documents.

        Output is a polars data frame with two columns named `d3_document_id` and
        `embedding`.
        """


@dataclass(kw_only=True)
class TFIDFEmbedder(Embedder):
    """
    Computes document embeddings with the TF-IDF algorithm.
    """

    tfidf_vectorizer: TfidfVectorizer

    def apply_tfidf_vectorizer(self, documents: Iterable[str]) -> np.ndarray:
        return TfidfVectorizer().fit_transform(documents).toarray()

    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        document_string = " ".join(document_tokens)
        tfidf_values = self.apply_tfidf_vectorizer([document_string])
        return tfidf_values[0].tolist()

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        token_strings = self.tokens_frame.to_pandas()["tokens"].str.join(" ")
        tfidf_values = self.apply_tfidf_vectorizer(token_strings)

        return self.tokens_frame.select("d3_document_id").with_columns(
            embedding=pl.Series(tfidf_values)
        )


@dataclass(kw_only=True)
class BM25Embedder(Embedder):
    """
    Computes document embeddings with the BM25 algorithm.
    """

    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        return bm25(document_tokens, self.tokens_frame["tokens"].to_list()).tolist()

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        with rich_progress_bar() as progress_bar:
            embeddings = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.compute_embedding_single_document)(row["tokens"])
                for row in progress_bar.track(
                    self.tokens_frame.iter_rows(named=True), total=len(self.tokens_frame)
                )
            )

        return self.tokens_frame.with_columns(embedding=pl.Series(embeddings)).drop("tokens")


@dataclass(kw_only=True)
class GensimEmbedder(Embedder):
    """
    Embedder for all models with a gensim interface. This includes Word2Vec, GloVe and
    FastText.
    """

    aggregation_strategy: AggregationStrategy = AggregationStrategy.mean
    keyed_vectors: KeyedVectors

    def token_embeddings_to_document_embedding(
        self, token_embeddings_per_document: np.ndarray
    ) -> np.ndarray:
        """
        Combines token embeddings per document by averaging each embedding dimension.

        Output has shape (n_dimensions,) with
        - n_dimensions: dimension of embedding space
        """
        if self.aggregation_strategy.is_mean:
            return np.mean(token_embeddings_per_document, axis=0)

        if self.aggregation_strategy.is_max:
            return np.max(token_embeddings_per_document, axis=0)

        raise ValueError(f"Aggregation strategy `{self.aggregation_strategy}` is not implemented.")

    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        token_embeddings = self.keyed_vectors[document_tokens]
        aggregated_embeddings = self.token_embeddings_to_document_embedding(token_embeddings)
        return aggregated_embeddings.tolist()

    def explode_tokens(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.explode("tokens")

    def filter_tokens(self, df: pl.LazyFrame, vocab: Collection) -> pl.LazyFrame:
        """
        Consider only tokens that are in the vocabulary of the pretrained embedding
        model.
        """
        return df.filter(pl.col("tokens").is_in(vocab))

    def add_token_embeddings(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Adds new column that contains the word embeddings for each token in the `tokens`
        column.
        """
        return df.with_columns(
            token_embedding=pl.col("tokens").apply(lambda token: self.keyed_vectors[token].tolist())
        )

    def add_unique_token_id(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Adds an additional unique token id to group by. This token id is different for
        two occurences of the same token within one document.

        The reason why grouping by the tokens itself is not sufficient is that groups of
        tokens with multiple occurences would be larger than groups of tokens with a
        single occurence.
        """
        return df.with_row_count("unique_token_id")

    def add_grouped_row_number(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        'Hack' to add a row number column for each group. Here each group is a unique
        combination of document and unique token.

        The added row number represents the dimension of the token embedding.
        """
        return df.with_columns(dummy_ones=pl.lit(1)).with_columns(
            dimension=pl.col("dummy_ones").cumsum().over(["d3_document_id", "unique_token_id"])
        )

    def average_token_embeddings_per_dimension(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Computes dimension-wise averages of all token embeddings within a document. The
        average has the same dimension as each token embedding and is independent of the
        number of tokens within a document.
        """
        return (
            df.pipe(self.add_unique_token_id)
            .explode("token_embedding")
            .pipe(self.add_grouped_row_number)
            .groupby(["d3_document_id", "dimension"], maintain_order=True)
            .agg(pl.col("token_embedding").mean())
        )

    def collapse_per_dimension_averages(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Collects the dimension wise averages of all tokens in a document from long
        format (one value per dimension) into a list.

        The output dataframe has a single row per document, where the `embedding` column
        contains lists with one entry per embedding dimension (and NOT per token within
        the document). This ensures that all lists have the same length which is equal
        to the embedding dimension of the pretrained embedding model.
        """
        return df.groupby("d3_document_id", maintain_order=True).agg(
            embedding=pl.col("token_embedding")
        )

    def compute_embeddings_frame_slice(self, tokens_frame_slice: TokensFrame) -> EmbeddingsFrame:
        """
        Computes embeddings for slices of the tokens frame.
        """
        vocab = self.keyed_vectors.key_to_index.keys()

        return (
            tokens_frame_slice.lazy()
            .pipe(self.explode_tokens)
            .pipe(self.filter_tokens, vocab=vocab)
            .pipe(self.add_token_embeddings)
            .pipe(self.average_token_embeddings_per_dimension)
            .pipe(self.collapse_per_dimension_averages)
            .collect()
        )

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        """
        Computes embeddings for all tokenized abstracts in the tokens frame.

        The output is a polars data frame with two columns named `d3_document_id` and
        `embedding`.
        """
        slice_size = 100
        num_rows = self.tokens_frame.height
        num_slices = num_rows // slice_size

        with rich_progress_bar() as progress_bar:
            return pl.concat(
                [
                    self.compute_embeddings_frame_slice(
                        self.tokens_frame.slice(next_index, slice_size)
                    )
                    for next_index in progress_bar.track(
                        range(0, num_rows, slice_size), total=num_slices
                    )
                ]
            )
