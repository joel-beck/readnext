from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from gensim.models import FastText, KeyedVectors
from transformers import BertModel

# do not import from .language_models to avoid circular imports
from readnext.modeling.language_models.tokenizer import (
    TokenIds,
    Tokens,
    TokensIdMapping,
    TokensMapping,
)
from readnext.utils import setup_progress_bar

Embedding: TypeAlias = np.ndarray
EmbeddingsMapping: TypeAlias = dict[int, Embedding]

KeywordAlgorithm: TypeAlias = Callable[[Tokens, Sequence[Tokens]], np.ndarray]


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
    The output dataframe has two columns: `document_id` and `embedding`.
    """
    return (
        pd.Series(embeddings_mapping, name="embedding")
        .to_frame()
        .rename_axis("document_id", axis="index")
        .reset_index(drop=False)
    )


@dataclass
class Embedder(ABC):
    """
    Abstract Base class for embedding models. All embedding models implement a
    `compute_embeddings_mapping` method.
    """

    @abstractmethod
    def compute_embeddings_mapping(self, tokens_mapping: TokensMapping) -> EmbeddingsMapping:
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


@dataclass
class TorchEmbedder(ABC):
    """
    Abstract Base class for pytorch embedding models. All embedding models implement a
    `compute_embeddings_mapping` method.
    """

    @abstractmethod
    def compute_embeddings_mapping(
        self, tokens_tensor_mapping: TokensIdMapping
    ) -> EmbeddingsMapping:
        """Computes a dictionary of document ids to document embeddings."""

    @staticmethod
    def aggregate_document_embeddings(
        document_embeddings: torch.Tensor,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
    ) -> torch.Tensor:
        """
        Combines word embeddings per document by averaging each embedding dimension.

        Output has shape (n_dimensions,) with
        - n_dimensions: dimension of embedding space
        """
        if aggregation_strategy.is_mean:
            return document_embeddings.mean(dim=1)

        if aggregation_strategy.is_max:
            return document_embeddings.max(dim=1)

        raise ValueError(f"Aggregation strategy `{aggregation_strategy}` is not implemented.")


@dataclass
class TFIDFEmbedder(Embedder):
    """
    Computes document embeddings with the TF-IDF or BM25 model.
    """

    keyword_algorithm: KeywordAlgorithm

    def compute_embeddings_single_document(
        self, document_tokens: Tokens, tokens_mapping: TokensMapping
    ) -> np.ndarray:
        return self.keyword_algorithm(document_tokens, list(tokens_mapping.values()))

    def compute_embeddings_mapping(self, tokens_mapping: TokensMapping) -> EmbeddingsMapping:
        """
        Takes a mapping of document ids to tokenized documents (each document is a list
        of strings) as input and computes the TF-IDF document embeddings.

        Output is a dictionary with document ids as keys and document embeddings as
        values. Each document embedding has shape (n_tokens,) with

        - n_tokens: number of tokens in the document
        """
        embeddings_mapping = {}

        with setup_progress_bar() as progress_bar:
            for document_id, document_tokens in progress_bar.track(
                tokens_mapping.items(),
                total=len(tokens_mapping),
                description="Computing TF-IDF Embeddings...",
            ):
                embeddings_mapping[document_id] = self.compute_embeddings_single_document(
                    document_tokens, tokens_mapping
                )

        return embeddings_mapping


@dataclass
class GensimEmbedder(Embedder):
    """
    Takes pretrained Word2Vec (`KeyedVectors` in gensim) or FastText model as input.
    """

    embedding_model: KeyedVectors | FastText
    aggregation_strategy: AggregationStrategy = AggregationStrategy.mean

    @abstractmethod
    def compute_embeddings_single_document(self, tokens: Tokens) -> np.ndarray:
        """
        Stacks all word embeddings of a single document vertically.

        Output has shape (n_tokens_input, n_dimensions) with
        - n_tokens_input: number of tokens in provided input document
        - n_dimensions: dimension of embedding space
        """

    def compute_embeddings_mapping(
        self,
        tokens_mapping: TokensMapping,
    ) -> EmbeddingsMapping:
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
            for document_id, tokens in progress_bar.track(
                tokens_mapping.items(),
                total=len(tokens_mapping),
                description="Computing Gensim Embeddings...",
            ):
                embeddings_mapping[document_id] = self.word_embeddings_to_document_embedding(
                    self.compute_embeddings_single_document(tokens), self.aggregation_strategy
                )

        return embeddings_mapping


class Word2VecEmbedder(GensimEmbedder):
    """Computes document embeddings with the Word2Vec model."""

    def __init__(self, embedding_model: KeyedVectors) -> None:
        super().__init__(embedding_model)

    def compute_embeddings_single_document(self, tokens: Tokens) -> np.ndarray:
        # exclude any individual unknown tokens
        return np.vstack(
            [self.embedding_model[token] for token in tokens if token in self.embedding_model]  # type: ignore # noqa: E501
        )


class FastTextEmbedder(GensimEmbedder):
    """Computes document embeddings with the FastText model."""

    def __init__(self, embedding_model: FastText) -> None:
        super().__init__(embedding_model)

    def compute_embeddings_single_document(self, tokens: Tokens) -> np.ndarray:
        return self.embedding_model.wv[tokens]  # type: ignore


@dataclass
class BERTEmbedder(TorchEmbedder):
    """
    Computes document embeddings with the BERT model. Takes a pretrained BERT model as
    input.
    """

    embedding_model: BertModel
    aggregation_strategy: AggregationStrategy = AggregationStrategy.mean

    def compute_embeddings_single_document(self, token_ids: TokenIds) -> Embedding:
        """
        Takes a tensor of a single tokenized document as input and computes the BERT
        token embeddings.

        Output has shape (n_documents_input, n_dimensions) with

        - n_documents_input: number of documents in provided input, corresponds to first
          dimension of
        `tokens_tensor` input
        - n_dimensions: dimension of embedding space
        """
        token_ids_tensor = torch.tensor([token_ids])

        # outputs is an ordered dictionary with keys `last_hidden_state` and `pooler_output`
        outputs = self.embedding_model(token_ids_tensor)

        # first element of outputs is the last hidden state of the [CLS] token
        # dimension: num_documents x num_tokens_per_document x embedding_dimension
        document_embeddings: torch.Tensor = outputs.last_hidden_state

        document_embedding = self.aggregate_document_embeddings(document_embeddings)

        return document_embedding.squeeze(0).detach().numpy()

    def compute_embeddings_mapping(
        self, tokens_tensor_mapping: TokensIdMapping
    ) -> EmbeddingsMapping:
        """
        Takes a tensor of tokenized documents as input and computes the BERT token
        embeddings.

        Output is a dictionary with document ids as keys and document embeddings as
        values. Each document embedding has shape (n_dimensions,) with

        - n_dimensions: dimension of the embedding space
        """
        embeddings_mapping = {}

        with setup_progress_bar() as progress_bar:
            for document_id, tokens_tensor in progress_bar.track(
                tokens_tensor_mapping.items(),
                total=len(tokens_tensor_mapping),
                description="Computing BERT embeddings...",
            ):
                embeddings_mapping[document_id] = self.compute_embeddings_single_document(
                    tokens_tensor
                )

        return embeddings_mapping
