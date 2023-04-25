from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias
from enum import Enum
import numpy as np
import pandas as pd
from gensim.models import FastText, KeyedVectors
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

# do not import from .language_models to avoid circular imports
from readnext.modeling.language_models.tokenizer import (
    DocumentsTokensTensor,
    DocumentsTokensTensorMapping,
    DocumentStringMapping,
    DocumentTokens,
    DocumentTokensMapping,
)
from readnext.utils import setup_progress_bar

EmbeddingModel: TypeAlias = TfidfVectorizer | BM25Okapi | KeyedVectors | FastText | BertModel

DocumentEmbeddings: TypeAlias = np.ndarray
DocumentEmbeddingsMapping: TypeAlias = dict[int, DocumentEmbeddings]


class AggregationStrategy(str, Enum):
    mean = "mean"
    max = "max"

    def __str__(self) -> str:
        return self.value

    @property
    def is_mean(self) -> bool:
        return self == self.mean

    @property
    def is_max(self) -> bool:
        return self == self.max


def embeddings_mapping_to_frame(embeddings_mapping: DocumentEmbeddingsMapping) -> pd.DataFrame:
    return (
        pd.Series(embeddings_mapping, name="embedding")
        .to_frame()
        .rename_axis("document_id", axis="index")
        .reset_index(drop=False)
    )


@dataclass
class Embedder(Protocol):
    embedding_model: EmbeddingModel

    def compute_embeddings_mapping(self) -> np.ndarray:
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

    def fit_embedding_model(self, tokens_mapping: DocumentStringMapping) -> None:
        self.embedding_model = self.embedding_model.fit(tokens_mapping.values())

    # one row per document, one column per token in vocabulary
    def compute_embeddings_mapping(
        self, tokens_mapping: DocumentStringMapping
    ) -> DocumentEmbeddingsMapping:
        """
        Takes a list of cleaned documents (list of strings, one string for each
        document) as input and computes the tfidf token embeddings.

        Output has shape (n_documents, n_tokens_vocab) with - n_documents: number of
        documents in provided input - n_tokens_vocab: number of tokens in vocabulary
        that was learned during training
        """
        self.embedding_model = self.embedding_model.fit(tokens_mapping.values())

        return {
            document_id: self.embedding_model.transform([document]).toarray()[0]
            for document_id, document in tokens_mapping.items()
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

    def compute_embedding_single_document(self, tokens: DocumentTokens) -> np.ndarray:
        """
        Takes a single tokenized document (list of strings, one string for each token)
        as input and computes the bm25 token embeddings.

        Output has shape (n_documents_training,) with
        - n_documents_training: number of documents in training corpus
        """
        return self.embedding_model.get_scores(tokens)  # type: ignore

    def compute_embeddings_mapping(
        self, tokens_mapping: DocumentTokensMapping
    ) -> DocumentEmbeddingsMapping:
        """
        Takes a list of tokenized documents (list of lists of strings, one list of
        strings for each document) as input and computes the bm25 token embeddings.

        Output has shape (n_documents_input, n_documents_training) with
        - n_documents_input: number of documents in provided input
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
    def compute_word_embeddings_per_document(self, tokens: DocumentTokens) -> np.ndarray:
        """
        Stacks all word embeddings of documents vertically.

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

        elif aggregation_strategy.is_max:
            return np.max(word_embeddings_per_document, axis=0)  # type: ignore

        raise ValueError(f"Aggregation strategy `{aggregation_strategy}` is not implemented.")

    def compute_embeddings_mapping(
        self,
        tokens_mapping: DocumentTokensMapping,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
    ) -> DocumentEmbeddingsMapping:
        """
        Takes a list of tokenized documents (list of lists of strings, one list of
        strings for each document) as input and computes the word2vec or fasttext token
        embeddings.

        Output has shape (n_documents_input, n_dimensions) with - n_documents_input:
        number of documents in provided input - n_dimensions: dimension of embedding
        space
        """
        return {
            document_id: self.compute_document_embedding(
                self.compute_word_embeddings_per_document(tokens), aggregation_strategy
            )
            for document_id, tokens in tokens_mapping.items()
        }


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

    def compute_embeddings_single_document(
        self,
        tokens_tensor: DocumentsTokensTensor,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
    ) -> DocumentEmbeddings:
        """
        Takes a tensor of a single tokenized document as input and computes the BERT
        token embeddings.

        Output has shape (n_documents_input, n_dimensions) with - n_documents_input:
        number of documents in provided input, corresponds to first dimension of
        `tokens_tensor` input - n_dimensions: dimension of embedding space
        """
        # outputs is an ordered dictionary with keys `last_hidden_state` and `pooler_output`
        # BertModel expects tensor of shape (batch_size, sequence_length), i.e. add
        # batch dimension to `tokens_tensor`
        outputs = self.embedding_model(tokens_tensor.unsqueeze(0))

        # first element of outputs is the last hidden state of the [CLS] token
        # dimension: num_documents x num_tokens_per_document x embedding_dimension
        document_embeddings: DocumentsTokensTensor = outputs.last_hidden_state

        if aggregation_strategy.is_mean:
            document_embedding = document_embeddings.mean(dim=1)
        elif aggregation_strategy.is_max:
            document_embedding = document_embeddings.max(dim=1)
        else:
            raise ValueError(f"Aggregation strategy `{aggregation_strategy}` is not implemented.")

        return document_embedding.squeeze(0).detach().numpy()  # type: ignore

    def compute_embeddings_mapping(
        self,
        tokens_tensor_mapping: DocumentsTokensTensorMapping,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
    ) -> DocumentEmbeddingsMapping:
        embeddings_mapping = {}

        with setup_progress_bar() as progress_bar:
            for document_id, tokens_tensor in progress_bar.track(
                tokens_tensor_mapping.items(), total=len(tokens_tensor_mapping)
            ):
                embeddings_mapping[document_id] = self.compute_embeddings_single_document(
                    tokens_tensor, aggregation_strategy
                )

        return embeddings_mapping
