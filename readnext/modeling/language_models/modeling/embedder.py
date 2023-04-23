from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
from gensim.models import FastText, KeyedVectors
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

from readnext.modeling.language_models.preprocessing import (
    DocumentsTokensList,
    DocumentsTokensString,
    DocumentsTokensTensor,
    DocumentTokens,
)

EmbeddingModel: TypeAlias = TfidfVectorizer | BM25Okapi | KeyedVectors | FastText | BertModel


def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    """Save document embeddings to disk"""
    np.save(path, embeddings)


def load_embeddings(path: Path) -> np.ndarray:
    """Load document embeddings from disk"""
    return np.load(path)  # type: ignore


@dataclass
class Embedder(Protocol):
    embedding_model: EmbeddingModel

    def compute_embeddings(self) -> np.ndarray:
        ...

    # def most_similar_to(
    #     self, document_index: int, n: int = 1
    # ) -> DocumentStatistics | list[DocumentStatistics]:
    #     """
    #     Return n most similar documents of training corpus for given input document. If
    #     n=1 return single DocumentStatistics object, otherwise return list of
    #     DocumentStatistics.
    #     """
    #     document_similarities = [
    #         DocumentStatistics(
    #             similarity=self.cosine_similarity(document_index, i),
    #             document_info=self.tokenizer.documents_info[i],
    #         )
    #         for i, _ in enumerate(self.embeddings)
    #         # exclude input document itself from list
    #         if i != document_index
    #     ]

    #     sorted_by_similarity = sorted(document_similarities, key=lambda x: -x.similarity)
    #     n_most_similar = sorted_by_similarity[:n]

    #     return n_most_similar[0] if n == 1 else n_most_similar


@dataclass
class TFIDFEmbedder:
    """
    Implements `Embedder` Protocol

    Takes TfidfVectorizer as input that has already been fitted on the training corpus.
    """

    embedding_model: TfidfVectorizer

    # one row per document, one column per token in vocabulary
    def compute_embeddings(self, tokens: DocumentsTokensString) -> np.ndarray:
        """
        Takes a list of cleaned documents (list of strings, one string for each
        document) as input and computes the tfidf token embeddings.

        Output has shape (n_documents, n_tokens_vocab) with - n_documents: number of
        documents in provided input - n_tokens_vocab: number of tokens in vocabulary
        that was learned during training
        """
        return self.embedding_model.fit_transform(tokens).toarray()  # type: ignore


# TODO: Can we compute document embeddings with bm25 just like with tfidf wothout
# providing a query? In other words, can we obtain a document-term-matrix with bm25 with
# dimensions (n_documents, n_tokens_vocab) like with tfidf? Right now we skip this step
# (violates protocol) and directly compute similarities without cosine similarity.
@dataclass
class BM25Embedder:
    """
    Implements `Embedder` Protocol

    Takes BM25 model as input that has already been fitted on the training corpus.
    """

    embedding_model: BM25Okapi

    def compute_embedding_single_document(self, tokens: DocumentTokens) -> np.ndarray:
        """
        Takes a single tokenized document (list of strings, one string for each token)
        as input and computes the bm25 token embeddings.

        Output has shape (n_documents_training,) with
        - n_documents_training: number of documents in training corpus
        """
        return self.embedding_model.get_scores(tokens)  # type: ignore

    def compute_embeddings(self, tokens_list: DocumentsTokensList) -> np.ndarray:
        """
        Takes a list of tokenized documents (list of lists of strings, one list of
        strings for each document) as input and computes the bm25 token embeddings.

        Output has shape (n_documents_input, n_documents_training) with
        - n_documents_input: number of documents in provided input
        - n_documents_training: number of documents in training corpus
        """
        return np.array([self.compute_embedding_single_document(tokens) for tokens in tokens_list])


@dataclass
class GensimEmbedder(ABC):
    """
    Implements `Embedder` Protocol

    Takes pretrained Word2Vec (`KeyedVectors` in gensim) or FastText model as input.
    """

    embedding_model: KeyedVectors | FastText

    @abstractmethod
    def compute_word_embeddings_per_document(self, tokens: DocumentTokens) -> np.ndarray:
        """
        Stacks all word embeddings of documents vertically.

        Output has shape (n_tokens_input, n_dimensions) with
        - n_tokens_input: number of tokens in provided input document
        - n_dimensions: dimension of embedding space
        """

    @staticmethod
    def compute_document_embedding(
        word_embeddings_per_document: np.ndarray, strategy: Literal["mean", "max"] = "mean"
    ) -> np.ndarray:
        """
        Combines word embeddings per document by averaging each embedding dimension.

        Output has shape (n_dimensions,) with
        - n_dimensions: dimension of embedding space
        """
        if strategy == "mean":
            return np.mean(word_embeddings_per_document, axis=0)  # type: ignore

        return np.max(word_embeddings_per_document, axis=0)  # type: ignore

    def compute_embeddings(
        self, tokens_list: DocumentsTokensList, strategy: Literal["mean", "max"] = "mean"
    ) -> np.ndarray:
        """
        Takes a list of tokenized documents (list of lists of strings, one list of
        strings for each document) as input and computes the fasttext token embeddings.

        Output has shape (n_documents_input, n_dimensions) with
        - n_documents_input: number of documents in provided input
        - n_dimensions: dimension of embedding space
        """
        document_embeddings_list = [
            self.compute_document_embedding(
                self.compute_word_embeddings_per_document(tokens), strategy
            )
            for tokens in tokens_list
        ]
        return np.vstack(document_embeddings_list)


class Word2VecEmbedder(GensimEmbedder):
    def __init__(self, embedding_model: KeyedVectors) -> None:
        super().__init__(embedding_model)

    def compute_word_embeddings_per_document(self, tokens: DocumentTokens) -> np.ndarray:
        # exclude any individual unknown tokens
        return np.vstack(
            [self.embedding_model[token] for token in tokens if token in self.embedding_model]  # type: ignore # noqa: E501
        )


class FastTextEmbedder(GensimEmbedder):
    def __init__(self, embedding_model: FastText) -> None:
        super().__init__(embedding_model)

    def compute_word_embeddings_per_document(self, tokens: DocumentTokens) -> np.ndarray:
        return self.embedding_model.wv[tokens]  # type: ignore


@dataclass
class BERTEmbedder:
    """
    Implements `Embedder` Protocol

    Takes pretrained BERT model as input.
    """

    embedding_model: BertModel

    def compute_embeddings(
        self, tokens_tensor: DocumentsTokensTensor, strategy: Literal["mean", "max"] = "mean"
    ) -> np.ndarray:
        """
        Takes a tensor of tokenized documents as input and computes the BERT token embeddings.

        Output has shape (n_documents_input, n_dimensions) with
        - n_documents_input: number of documents in provided input, corresponds to first
        dimension of `tokens_tensor` input
        - n_dimensions: dimension of embedding space
        """
        # outputs is an ordered dictionary with keys `last_hidden_state` and `pooler_output`
        outputs = self.embedding_model(tokens_tensor)

        # first element of outputs is the last hidden state of the [CLS] token
        # dimension: num_documents x num_tokens_per_document x embedding_dimension
        all_document_embeddings = outputs.last_hidden_state

        if strategy == "mean":
            document_embeddings = all_document_embeddings.mean(dim=1)
        else:
            document_embeddings = all_document_embeddings.max(dim=1)

        return document_embeddings.detach().numpy()  # type: ignore
