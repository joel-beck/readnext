from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import polars as pl
from gensim.models.keyedvectors import KeyedVectors
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

from readnext.data.data_split import DataSplitIndices
from readnext.modeling.language_models.bm25 import bm25
from readnext.utils.aliases import Embedding, EmbeddingsFrame, Tokens, TokensFrame
from readnext.utils.progress_bar import rich_progress_bar
from readnext.utils.slicing import concatenate_sliced_dataframes


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

    def __post_init__(self) -> None:
        self.split_tokens_frame()
        self.set_token_strings()

    def split_tokens_frame(self) -> None:
        """
        Split the full tokens frame into training, validation and test sets.
        """
        data_split_indices = DataSplitIndices.from_frames()

        self.tokens_frame_train = self.tokens_frame.filter(
            pl.col("d3_document_id").is_in(data_split_indices.train)
        )
        self.tokens_frame_validation = self.tokens_frame.filter(
            pl.col("d3_document_id").is_in(data_split_indices.validation)
        )
        self.tokens_frame_test = self.tokens_frame.filter(
            pl.col("d3_document_id").is_in(data_split_indices.test)
        )

    def convert_tokens_to_token_strings(self, tokens_frame: TokensFrame) -> pd.Series:
        """
        Convert tokens (list of strings) to strings that can be processed by the TF-IDF
        vectorizer.
        """
        return tokens_frame.to_pandas()["tokens"].str.join(" ")

    def set_token_strings(self) -> None:
        """
        Convert all tokens in the `tokens` column of the tokens frames to strings.
        """
        self.token_strings_train = self.convert_tokens_to_token_strings(self.tokens_frame_train)
        self.token_strings_validation = self.convert_tokens_to_token_strings(
            self.tokens_frame_validation
        )
        self.token_strings_test = self.convert_tokens_to_token_strings(self.tokens_frame_test)

    def fit_tfidf_vectorizer(self, training_documents: pd.Series) -> None:
        """
        Fit TF-IDF vectorizer on the entire training corpus to learn the vocabulary.
        """
        self.tfidf_vectorizer.fit(training_documents)

    def apply_tfidf_vectorizer(self, documents: pd.Series | list[str]) -> np.ndarray:
        """
        Use the fitted TF-IDF vectorizer for inference on a single or multiple documents.
        """
        return self.tfidf_vectorizer.transform(documents).toarray()

    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        """
        Fit the TF-IDF vectorizer on the training corpus and use it to compute the
        document embedding for a single test document.
        """
        document_string = " ".join(document_tokens)
        self.fit_tfidf_vectorizer(self.token_strings_train)
        tfidf_embedding = self.apply_tfidf_vectorizer([document_string])
        return tfidf_embedding[0].tolist()

    @staticmethod
    def enframe_embeddings(
        tokens_frame_subset: TokensFrame, embeddings_subset: np.ndarray
    ) -> EmbeddingsFrame:
        """
        Add the matching indices from the tokens frame to the embeddings array.
        """
        return tokens_frame_subset.select("d3_document_id").with_columns(
            embedding=pl.Series(embeddings_subset.tolist())
        )

    def concatenate_embeddings(
        self,
        embeddings_train: np.ndarray,
        embeddings_validation: np.ndarray,
        embeddings_test: np.ndarray,
    ) -> pl.DataFrame:
        """
        Concatenate the embeddings for the training, validation and test sets. The
        training embeddings were used for training, the validation embeddings and test
        embeddings are computed from the fitted TF-IDF vectorizer.
        """
        return pl.concat(
            [
                self.enframe_embeddings(self.tokens_frame_train, embeddings_train),
                self.enframe_embeddings(self.tokens_frame_validation, embeddings_validation),
                self.enframe_embeddings(self.tokens_frame_test, embeddings_test),
            ]
        )

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        """
        Compute the document embeddings for all documents in the training, validation
        and test sets.
        """
        self.fit_tfidf_vectorizer(self.token_strings_train)
        embeddings_train = self.apply_tfidf_vectorizer(self.token_strings_train)
        embeddings_validation = self.apply_tfidf_vectorizer(self.token_strings_validation)
        embeddings_test = self.apply_tfidf_vectorizer(self.token_strings_test)

        return self.concatenate_embeddings(embeddings_train, embeddings_validation, embeddings_test)


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
        return concatenate_sliced_dataframes(
            df=self.tokens_frame,
            slice_function=self.compute_embeddings_frame_slice,
            slice_size=100,
            progress_bar_description=f"{self.keyed_vectors.__class__.__name__}:",
        )
