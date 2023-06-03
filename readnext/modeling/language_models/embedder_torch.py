from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from tqdm import tqdm

from readnext.modeling.language_models.embedder import AggregationStrategy
from readnext.utils import (
    BertModelProtocol,
    Embedding,
    EmbeddingsFrame,
    LongformerModelProtocol,
    TokenIds,
    TokenIdsFrame,
    tqdm_progress_bar_wrapper,
)

TTorchModel = TypeVar("TTorchModel", bound=BertModelProtocol | LongformerModelProtocol)


@dataclass(kw_only=True)
class TorchEmbedder(ABC, Generic[TTorchModel]):
    """
    Abstract Base class for pytorch embedding models. All embedding models implement
    `compute_embedding_single_document()` and `compute_embeddings_frame()` methods.
    """

    token_ids_frame: TokenIdsFrame
    torch_model: TTorchModel
    aggregation_strategy: AggregationStrategy = AggregationStrategy.mean

    @staticmethod
    def aggregate_document_embeddings(
        document_embeddings: torch.Tensor,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
    ) -> torch.Tensor:
        """
        Collapses first dimension of document embeddings tensor (number of tokens per
        document) to a single document embedding.
        """
        if aggregation_strategy.is_mean:
            return document_embeddings.mean(dim=1)

        if aggregation_strategy.is_max:
            return document_embeddings.max(dim=1)[0]  # Only return the max values, not the indices

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

        return document_embedding.squeeze(0).detach().tolist()

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        """
        Takes a tensor of tokenized documents as input and computes the BERT or
        Longformer token embeddings.

        Output is a polars data frame with two columns named `d3_document_id` and
        `embedding`.
        """
        with tqdm(total=len(self.token_ids_frame)) as progress_bar:
            embeddings_frame = self.token_ids_frame.with_columns(
                embedding=self.token_ids_frame["token_ids"].apply(
                    tqdm_progress_bar_wrapper(
                        progress_bar,
                        # each row is implicitly converted to a polars Series from a
                        # Python list during `apply()`
                        lambda row: self.compute_embedding_single_document(row.to_list()),
                    )
                )
            )

        return embeddings_frame.drop("token_ids")


@dataclass(kw_only=True)
class BERTEmbedder(TorchEmbedder):
    """
    Computes document embeddings with the BERT model. Takes a pretrained BERT model as
    input.
    """

    torch_model: BertModelProtocol


@dataclass(kw_only=True)
class LongformerEmbedder(TorchEmbedder):
    """
    Computes document embeddings with the Longformer model. Takes a pretrained Longformer model as
    input.
    """

    torch_model: LongformerModelProtocol
