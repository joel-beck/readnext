from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import numpy as np
from gensim.models.fasttext import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

from readnext.modeling.language_models import (
    DocumentsTokensList,
    DocumentsTokensString,
    DocumentsTokensTensor,
    DocumentTokens,
)

EmbeddingModel: TypeAlias = TfidfVectorizer | FastText | BertModel


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    """Save document embeddings to disk"""
    np.save(path, embeddings)


def load_embeddings(path: Path) -> np.ndarray:
    """Load document embeddings from disk"""
    return np.load(path)  # type: ignore


@dataclass
class Embedder(Protocol):
    embedding_model: EmbeddingModel
    # embeddings: np.ndarray = field(default=np.ndarray([]))

    def compute_embeddings(self) -> np.ndarray:
        ...

    # def add_embeddings(self) -> None:
    #     """Store document embeddings as a property to the class"""
    #     self.embeddings = self._compute_embeddings()

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
    """Implements `Embedder` Protocol"""

    embedding_model: TfidfVectorizer
    # embeddings: np.ndarray = field(default=np.ndarray([]))

    # one row per document, one column per token in vocabulary
    def compute_embeddings(self, tokens: DocumentsTokensString) -> np.ndarray:
        return self.embedding_model.fit_transform(tokens).toarray()  # type: ignore


@dataclass
class FastTextEmbedder:
    """Implements `Embedder` Protocol"""

    embedding_model: FastText
    # embeddings: np.ndarray = field(default=np.ndarray([]))

    def compute_word_embeddings_per_document(self, tokens: DocumentTokens) -> np.ndarray:
        """
        Stacks all word embeddings of documents vertically. Output has shape (n_tokens,
        n_dimensions).
        """
        return self.embedding_model.wv[tokens]  # type: ignore

    @staticmethod
    def compute_document_embedding(
        word_embeddings_per_document: np.ndarray, strategy: Literal["mean", "max"] = "mean"
    ) -> np.ndarray:
        """
        Combines word embeddings per document by averaging each embedding dimension.
        Output has shape (n_dimensions,).
        """
        if strategy == "mean":
            return np.mean(word_embeddings_per_document, axis=0)  # type: ignore

        return np.max(word_embeddings_per_document, axis=0)  # type: ignore

    def compute_embeddings(
        self, tokens_list: DocumentsTokensList, strategy: Literal["mean", "max"] = "mean"
    ) -> np.ndarray:
        document_embeddings_list = [
            self.compute_document_embedding(
                self.compute_word_embeddings_per_document(tokens), strategy
            )
            for tokens in tokens_list
        ]
        return np.vstack(document_embeddings_list)


@dataclass
class BERTEmbedder:
    """Implements `Embedder` Protocol"""

    embedding_model: BertModel
    # embeddings: np.ndarray = field(default=np.ndarray([]))

    def compute_embeddings(
        self, tokens_tensor: DocumentsTokensTensor, strategy: Literal["mean", "max"] = "mean"
    ) -> np.ndarray:
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
