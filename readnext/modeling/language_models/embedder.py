from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# do not import from .language_models to avoid circular imports
from readnext.modeling.language_models.tokenizer_list import Tokens, TokensMapping
from readnext.utils import (
    EmbeddingsMapping,
    FastTextModelProtocol,
    KeywordAlgorithm,
    Word2VecModelProtocol,
    setup_progress_bar,
)


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


def embeddings_mapping_to_frame(embeddings_mapping: EmbeddingsMapping) -> pd.DataFrame:
    """
    Converts a dictionary of document ids to document embeddings to a pandas DataFrame.
    The output dataframe has one column named `embedding` and the index named
    `document_id`.
    """
    return (
        pd.Series(embeddings_mapping, name="embedding")
        .to_frame()
        .rename_axis("document_id", axis="index")
    )


@dataclass(kw_only=True)
class Embedder(ABC):
    """
    Abstract Base class for embedding models. All embedding models implement a
    `compute_embeddings_mapping` method.
    """

    @abstractmethod
    def compute_embedding_single_document(self, document_tokens: Tokens) -> np.ndarray:
        """Computes a single document embedding from a tokenized document."""

    @abstractmethod
    def compute_embeddings_mapping(self) -> EmbeddingsMapping:
        """Computes a dictionary of document ids to document embeddings."""

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
    Computes document embeddings with the TF-IDF or BM25 model.
    """

    tokens_mapping: TokensMapping
    keyword_algorithm: KeywordAlgorithm

    def compute_embedding_single_document(self, document_tokens: Tokens) -> np.ndarray:
        return self.keyword_algorithm(document_tokens, list(self.tokens_mapping.values()))

    def compute_embeddings_mapping(self) -> EmbeddingsMapping:
        """
        Takes a mapping of document ids to tokenized documents (each document is a list
        of strings) as input and computes the TF-IDF document embeddings.

        Output is a dictionary with document ids as keys and document embeddings as
        values. Each document embedding has shape (n_tokens,) with

        - n_tokens: number of tokens in the document
        """
        embeddings_mapping = {}

        with setup_progress_bar() as progress_bar:
            for d3_document_id, document_tokens in progress_bar.track(
                self.tokens_mapping.items(),
                total=len(self.tokens_mapping),
                description=f"{self.__class__.__name__}:",
            ):
                embeddings_mapping[d3_document_id] = self.compute_embedding_single_document(
                    document_tokens
                )

        return embeddings_mapping


@dataclass(kw_only=True)
class GensimEmbedder(Embedder):
    """
    Takes pretrained Word2Vec (`KeyedVectors` in gensim) or FastText model as input.
    """

    tokens_mapping: TokensMapping
    aggregation_strategy: AggregationStrategy = AggregationStrategy.mean

    @abstractmethod
    def compute_embedding_single_document(self, document_tokens: Tokens) -> np.ndarray:
        """
        Stacks all word embeddings of a single document vertically.

        Output has shape (n_tokens_input, n_dimensions) with
        - n_tokens_input: number of tokens in provided input document
        - n_dimensions: dimension of embedding space
        """

    def compute_embeddings_mapping(self) -> EmbeddingsMapping:
        """
        Takes a list of tokenized documents (list of lists of strings, one list of
        strings for each document) as input and computes the word2vec or fasttext token
        embeddings.

        Output is a dictionary with document ids as keys and document embeddings as
        values. Each document embedding has shape (n_dimensions,) with

        - n_dimensions: dimension of the embedding space
        """

        embeddings_mapping = {}

        with setup_progress_bar() as progress_bar:
            for d3_document_id, tokens in progress_bar.track(
                self.tokens_mapping.items(),
                total=len(self.tokens_mapping),
                description=f"{self.__class__.__name__}:",
            ):
                embeddings_mapping[d3_document_id] = self.compute_embedding_single_document(tokens)

        return embeddings_mapping


class Word2VecEmbedder(GensimEmbedder):
    """Computes document embeddings with the Word2Vec model."""

    def __init__(
        self, tokens_mapping: TokensMapping, embedding_model: Word2VecModelProtocol
    ) -> None:
        super().__init__(tokens_mapping=tokens_mapping)
        self.embedding_model = embedding_model

    def compute_embedding_single_document(self, document_tokens: Tokens) -> np.ndarray:
        # exclude any individual unknown tokens
        stacked_word_embeddings = np.vstack(
            [self.embedding_model[token] for token in document_tokens if token in self.embedding_model]  # type: ignore # noqa: E501
        )

        return self.word_embeddings_to_document_embedding(
            stacked_word_embeddings, self.aggregation_strategy
        )


class FastTextEmbedder(GensimEmbedder):
    """Computes document embeddings with the FastText model."""

    def __init__(
        self, tokens_mapping: TokensMapping, embedding_model: FastTextModelProtocol
    ) -> None:
        super().__init__(tokens_mapping=tokens_mapping)
        self.embedding_model = embedding_model

    def compute_embedding_single_document(self, document_tokens: Tokens) -> np.ndarray:
        return self.word_embeddings_to_document_embedding(
            self.embedding_model.wv[document_tokens],
            self.aggregation_strategy,
        )  # type: ignore
