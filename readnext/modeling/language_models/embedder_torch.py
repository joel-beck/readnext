from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from transformers import BertModel, LongformerModel

from readnext.modeling.language_models.embedder import (
    AggregationStrategy,
    Embedding,
    EmbeddingsMapping,
)

# do not import from .language_models to avoid circular imports
from readnext.modeling.language_models.tokenizer_tensor import TokenIds, TokensIdMapping
from readnext.utils import setup_progress_bar

TTorchModel = TypeVar("TTorchModel", bound=BertModel | LongformerModel)


@dataclass
class TorchEmbedder(ABC, Generic[TTorchModel]):
    """
    Abstract Base class for pytorch embedding models. All embedding models implement a
    `compute_embeddings_mapping` method.
    """

    torch_model: TTorchModel
    tokens_tensor_mapping: TokensIdMapping
    aggregation_strategy: AggregationStrategy = AggregationStrategy.mean

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

    def compute_embedding_single_document(self, token_ids: TokenIds) -> Embedding:
        """
        Takes a tensor of a single tokenized document as input and computes the BERT or
        Longformer token embedding.

        Output has shape (n_documents_input, n_dimensions) with

        - n_documents_input: number of documents in provided input, corresponds to first
          dimension of
        `tokens_tensor` input - n_dimensions: dimension of embedding space
        """
        token_ids_tensor = torch.tensor([token_ids])

        # outputs is an ordered dictionary with keys `last_hidden_state` and `pooler_output`
        outputs = self.torch_model(token_ids_tensor)

        # first element of outputs is the last hidden state of the [CLS] token
        # dimension: num_documents x num_tokens_per_document x embedding_dimension
        document_embeddings: torch.Tensor = outputs.last_hidden_state
        document_embedding = self.aggregate_document_embeddings(document_embeddings)

        return document_embedding.squeeze(0).detach().numpy()

    def compute_embeddings_mapping(self) -> EmbeddingsMapping:
        """
        Takes a tensor of tokenized documents as input and computes the BERT or
        Longformer token embeddings.

        Output is a dictionary with document ids as keys and document embeddings as
        values. Each document embedding has shape (n_dimensions,) with

        - n_dimensions: dimension of the embedding space
        """
        embeddings_mapping = {}

        with setup_progress_bar() as progress_bar:
            for document_id, tokens_tensor in progress_bar.track(
                self.tokens_tensor_mapping.items(),
                total=len(self.tokens_tensor_mapping),
                description=f"{self.__class__.__name__}:",
            ):
                embeddings_mapping[document_id] = self.compute_embedding_single_document(
                    tokens_tensor
                )

        return embeddings_mapping


@dataclass
class BERTEmbedder(TorchEmbedder):
    """
    Computes document embeddings with the BERT model. Takes a pretrained BERT model as
    input.
    """

    torch_model: BertModel


@dataclass
class LongformerEmbedder(TorchEmbedder):
    """
    Computes document embeddings with the Longformer model. Takes a pretrained Longformer model as
    input.
    """

    torch_model: LongformerModel