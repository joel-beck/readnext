from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

from readnext.modeling.language_models.bm25 import bm25
from readnext.utils.aliases import Embedding, EmbeddingsFrame, Tokens, TokensFrame
from readnext.utils.progress_bar import rich_progress_bar
from readnext.utils.protocols import FastTextModelProtocol, Word2VecModelProtocol


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
    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        """
        Takes a tokens frame with the two columns `d3_document_id` and `tokens` as input
        and computes the abstract embeddings for all documents.

        Output is a polars data frame with two columns named `d3_document_id` and
        `embedding`.
        """
        # with rich_progress_bar() as progress_bar:
        #     embeddings = Parallel(n_jobs=-1, prefer="threads")(
        #         delayed(self.compute_embedding_single_document)(row["tokens"])
        #         for row in progress_bar.track(
        #             self.tokens_frame.iter_rows(named=True), total=len(self.tokens_frame)
        #         )
        #     )

        # return self.tokens_frame.with_columns(embedding=pl.Series(embeddings)).drop("tokens")

    @staticmethod
    def word_embeddings_to_document_embedding(
        word_embeddings_per_document: np.ndarray, aggregation_strategy: AggregationStrategy
    ) -> np.ndarray:
        """
        Combines word embeddings per document by averaging each embedding dimension.

        Output has shape (n_dimensions,) with
        - n_dimensions: dimension of embedding space
        """
        if aggregation_strategy.is_mean:
            return np.mean(word_embeddings_per_document, axis=0)

        if aggregation_strategy.is_max:
            return np.max(word_embeddings_per_document, axis=0)

        raise ValueError(f"Aggregation strategy `{aggregation_strategy}` is not implemented.")


@dataclass(kw_only=True)
class TFIDFEmbedder(Embedder):
    """
    Computes document embeddings with the TF-IDF algorithm.
    """

    tfidf_vectorizer: TfidfVectorizer

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        token_strings = self.tokens_frame.to_pandas()["tokens"].str.join(" ")
        tfidf_values = self.tfidf_vectorizer.fit_transform(token_strings).toarray()

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
    Takes pretrained Word2Vec (`KeyedVectors` in gensim) or FastText model as input.
    """

    aggregation_strategy: AggregationStrategy = AggregationStrategy.mean

    @abstractmethod
    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        """
        Stacks all word embeddings of a single document vertically.

        Output has shape (n_tokens_input, n_dimensions) with
        - n_tokens_input: number of tokens in provided input document
        - n_dimensions: dimension of embedding space
        """


class Word2VecEmbedder(GensimEmbedder):
    """Computes document embeddings with the Word2Vec model."""

    def __init__(self, tokens_frame: TokensFrame, embedding_model: Word2VecModelProtocol) -> None:
        super().__init__(tokens_frame=tokens_frame)
        self.embedding_model = embedding_model

    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        # exclude any individual unknown tokens
        stacked_word_embeddings = np.vstack(
            [self.embedding_model[token] for token in document_tokens if token in self.embedding_model]  # type: ignore # noqa: E501
        )

        return self.word_embeddings_to_document_embedding(
            stacked_word_embeddings, self.aggregation_strategy
        ).tolist()


class FastTextEmbedder(GensimEmbedder):
    """Computes document embeddings with the FastText model."""

    def __init__(self, tokens_frame: TokensFrame, embedding_model: FastTextModelProtocol) -> None:
        super().__init__(tokens_frame=tokens_frame)
        self.embedding_model = embedding_model

    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        return self.word_embeddings_to_document_embedding(
            self.embedding_model.wv[document_tokens],
            self.aggregation_strategy,
        ).tolist()
