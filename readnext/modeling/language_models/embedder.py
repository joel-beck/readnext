from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

import numpy as np
import pandas as pd
import torch
from gensim.models import FastText, KeyedVectors
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

# do not import from .language_models to avoid circular imports
from readnext.modeling.language_models.tokenizer import (
    StringMapping,
    TokenIds,
    Tokens,
    TokensIdMapping,
    TokensMapping,
)
from readnext.utils import setup_progress_bar

EmbeddingModel: TypeAlias = TfidfVectorizer | BM25Okapi | KeyedVectors | FastText | BertModel

Embedding: TypeAlias = np.ndarray
EmbeddingsMapping: TypeAlias = dict[int, Embedding]


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
class Embedder(Protocol):
    """
    Protocol for embedding models. All embedding models take an embedding model as input
    and implement a `compute_embeddings_mapping` method.
    """

    embedding_model: EmbeddingModel

    def compute_embeddings_mapping(self) -> EmbeddingsMapping:
        """Computes a dictionary of document ids to document embeddings."""
        ...


@dataclass
class TFIDFEmbedder:
    """
    Implements `Embedder` Protocol

    Computes document embeddings with the TF-IDF model.
    """

    embedding_model: TfidfVectorizer

    def fit_embedding_model(self, tokens_mapping: StringMapping) -> None:
        self.embedding_model = self.embedding_model.fit(tokens_mapping.values())

    # one row per document, one column per token in vocabulary
    def compute_embeddings_mapping(self, tokens_mapping: StringMapping) -> EmbeddingsMapping:
        """
        Takes a list of cleaned documents (list of strings, one string for each
        document) as input and computes the tfidf token embeddings.

        Output is a dictionary with document ids as keys and document embeddings as
        values. Each document embedding has shape (n_tokens_vocab,) with

        - n_tokens_vocab: number of tokens in the vocabulary that was learned during
          training
        """
        self.embedding_model = self.embedding_model.fit(tokens_mapping.values())

        with setup_progress_bar() as progress_bar:
            return {
                document_id: self.embedding_model.transform([document]).toarray()[0]
                for document_id, document in progress_bar.track(
                    tokens_mapping.items(), total=len(tokens_mapping)
                )
            }


# TODO: Can we compute document embeddings with bm25 just like with tfidf wothout
# providing a query? In other words, can we obtain a document-term-matrix with bm25 with
# dimensions (n_documents, n_tokens_vocab) like with tfidf? Right now we skip this step
# (violates protocol) and directly compute similarities without cosine similarity.
# Possible Solution: Manually implement bm25 as done here:
# https://www.pinecone.io/learn/semantic-search/
@dataclass
class BM25Embedder:
    """
    Implements `Embedder` Protocol

    Takes BM25 model as input that has already been fitted on the training corpus.
    """

    embedding_model: BM25Okapi

    def compute_embedding_single_document(self, tokens: Tokens) -> np.ndarray:
        """
        Takes a single tokenized document (list of strings, one string for each token)
        as input and computes the bm25 token embeddings.

        Output has shape (n_documents_training,) with

        - n_documents_training: number of documents in training corpus
        """
        return self.embedding_model.get_scores(tokens)  # type: ignore

    def compute_embeddings_mapping(self, tokens_mapping: TokensMapping) -> EmbeddingsMapping:
        """
        Takes a list of tokenized documents (list of lists of strings, one list of
        strings for each document) as input and computes the bm25 token embeddings.

        Output is a dictionary with document ids as keys and document embeddings as
        values. Each document embedding has shape (n_documents_training,) with

        - n_documents_training: number of documents in training corpus
        """
        return {
            document_id: self.compute_embedding_single_document(tokens)
            for document_id, tokens in tokens_mapping.items()
        }


@dataclass
class GensimEmbedder(ABC):
    """
    Implements `Embedder` Protocol

    Takes pretrained Word2Vec (`KeyedVectors` in gensim) or FastText model as input.
    """

    embedding_model: KeyedVectors | FastText

    @abstractmethod
    def compute_word_embeddings_per_document(self, tokens: Tokens) -> np.ndarray:
        """
        Stacks all word embeddings of a single document vertically.

        Output has shape (n_tokens_input, n_dimensions) with
        - n_tokens_input: number of tokens in provided input document
        - n_dimensions: dimension of embedding space
        """

    @staticmethod
    def compute_document_embedding(
        word_embeddings_per_document: np.ndarray,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
    ) -> np.ndarray:
        """
        Combines word embeddings per document by averaging each embedding dimension.

        Output has shape (n_dimensions,) with
        - n_dimensions: dimension of embedding space
        """
        if aggregation_strategy.is_mean:
            return np.mean(word_embeddings_per_document, axis=0)  # type: ignore

        if aggregation_strategy.is_max:
            return np.max(word_embeddings_per_document, axis=0)  # type: ignore

        raise ValueError(f"Aggregation strategy `{aggregation_strategy}` is not implemented.")

    def compute_embeddings_mapping(
        self,
        tokens_mapping: TokensMapping,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
    ) -> EmbeddingsMapping:
        """
        Takes a list of tokenized documents (list of lists of strings, one list of
        strings for each document) as input and computes the word2vec or fasttext token
        embeddings.

        Output is a dictionary with document ids as keys and document embeddings as
        values. Each document embedding has shape (n_dimensions,) with

        - n_dimensions: dimension of the embedding space
        """

        with setup_progress_bar() as progress_bar:
            return {
                document_id: self.compute_document_embedding(
                    self.compute_word_embeddings_per_document(tokens), aggregation_strategy
                )
                for document_id, tokens in progress_bar.track(
                    tokens_mapping.items(), total=len(tokens_mapping)
                )
            }


class Word2VecEmbedder(GensimEmbedder):
    """Computes document embeddings with the Word2Vec model."""

    def __init__(self, embedding_model: KeyedVectors) -> None:
        super().__init__(embedding_model)

    def compute_word_embeddings_per_document(self, tokens: Tokens) -> np.ndarray:
        # exclude any individual unknown tokens
        return np.vstack(
            [self.embedding_model[token] for token in tokens if token in self.embedding_model]  # type: ignore # noqa: E501
        )


class FastTextEmbedder(GensimEmbedder):
    """Computes document embeddings with the FastText model."""

    def __init__(self, embedding_model: FastText) -> None:
        super().__init__(embedding_model)

    def compute_word_embeddings_per_document(self, tokens: Tokens) -> np.ndarray:
        return self.embedding_model.wv[tokens]  # type: ignore


@dataclass
class BERTEmbedder:
    """
    Implements `Embedder` Protocol

    Takes a pretrained BERT model as input.
    """

    embedding_model: BertModel

    def compute_embeddings_single_document(
        self,
        token_ids: TokenIds,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
    ) -> Embedding:
        """
        Takes a tensor of a single tokenized document as input and computes the BERT
        token embeddings.

        Output has shape (n_documents_input, n_dimensions) with

        - n_documents_input: number of documents in provided input, corresponds to first
          dimension of
        `tokens_tensor` input
        - n_dimensions: dimension of embedding space
        """
        # outputs is an ordered dictionary with keys `last_hidden_state` and `pooler_output`
        outputs = self.embedding_model(torch.tensor([token_ids]))

        # first element of outputs is the last hidden state of the [CLS] token
        # dimension: num_documents x num_tokens_per_document x embedding_dimension
        document_embeddings: torch.Tensor = outputs.last_hidden_state

        if aggregation_strategy.is_mean:
            document_embedding = document_embeddings.mean(dim=1)
        elif aggregation_strategy.is_max:
            document_embedding = document_embeddings.max(dim=1)
        else:
            raise ValueError(f"Aggregation strategy `{aggregation_strategy}` is not implemented.")

        return document_embedding.squeeze(0).detach().numpy()  # type: ignore

    def compute_embeddings_mapping(
        self,
        tokens_tensor_mapping: TokensIdMapping,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
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
                tokens_tensor_mapping.items(), total=len(tokens_tensor_mapping)
            ):
                embeddings_mapping[document_id] = self.compute_embeddings_single_document(
                    tokens_tensor, aggregation_strategy
                )

        return embeddings_mapping
