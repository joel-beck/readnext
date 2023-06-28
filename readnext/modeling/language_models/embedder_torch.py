from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import polars as pl
import torch
from transformers import BertModel, LongformerModel

from readnext.modeling.language_models.embedder import AggregationStrategy
from readnext.utils.aliases import Embedding, EmbeddingsFrame, TokenIds, TokenIdsFrame
from readnext.utils.progress_bar import rich_progress_bar
from readnext.utils.torch_device import get_torch_device

TTorchModel = TypeVar("TTorchModel", bound=BertModel | LongformerModel)


@dataclass(kw_only=True)
class TorchEmbedder(ABC, Generic[TTorchModel]):
    """
    Abstract Base class for pytorch embedding models. All embedding models implement
    `compute_embedding_single_document()` and `compute_embeddings_frame()` methods.
    """

    token_ids_frame: TokenIdsFrame
    torch_model: TTorchModel
    device: torch.device = field(default_factory=get_torch_device)
    aggregation_strategy: AggregationStrategy = AggregationStrategy.mean

    def aggregate_document_embeddings(self, document_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Collapses first dimension of document embeddings tensor (number of tokens per
        document) to a single document embedding.
        """
        if self.aggregation_strategy.is_mean:
            return torch.mean(document_embeddings, dim=1)

        if self.aggregation_strategy.is_max:
            # Index 0 to only return the max values, not the indices
            return torch.max(document_embeddings, dim=1)[0]

        raise ValueError(f"Aggregation strategy `{self.aggregation_strategy}` is not implemented.")

    def compute_embeddings(self, tokenized_documents: list[TokenIds]) -> torch.Tensor:
        """
        Computes embeddings for multiple tokenized documents.

        Output is a Tensor with shape (n_documents_input, n_dimensions) with

        - n_documents_input: number of documents in provided input
        - n_dimensions: dimension of embedding space
        """
        bert_model = self.torch_model.to(self.device)

        with torch.no_grad():
            bert_model.eval()

            # outputs is an ordered dictionary with keys `last_hidden_state` and `pooler_output`
            outputs = (
                bert_model(torch.tensor(tokenized_documents, device=self.device))
                .last_hidden_state.detach()
                .cpu()
            )

            return self.aggregate_document_embeddings(outputs)

    def compute_embedding_single_document(self, document_token_ids: TokenIds) -> Embedding:
        """
        Compute the abstract embedding for a single document.
        """
        return self.compute_embeddings([document_token_ids])[0].tolist()

    def compute_embeddings_frame_slice(
        self, token_ids_frame_slice: TokenIdsFrame
    ) -> EmbeddingsFrame:
        """
        Computes abstract embeddings for slices of the token ids frame.
        """
        token_ids = token_ids_frame_slice["token_ids"].to_list()
        outputs = self.compute_embeddings(token_ids)

        return token_ids_frame_slice.select("d3_document_id").with_columns(
            embedding=pl.Series(outputs.tolist())
        )

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        """
        Computes embeddings for all tokenized abstracts in the token ids frame.

        The output is a polars data frame with two columns named `d3_document_id` and
        `embedding`.
        """
        slice_size = 5
        num_rows = self.token_ids_frame.height
        num_slices = num_rows // slice_size

        with rich_progress_bar() as progress_bar:
            return pl.concat(
                [
                    self.compute_embeddings_frame_slice(
                        self.token_ids_frame.slice(next_index, slice_size)
                    )
                    for next_index in progress_bar.track(
                        range(0, num_rows, slice_size),
                        total=num_slices,
                        description=f"{self.torch_model.__class__.__name__}:",
                    )
                ]
            )


@dataclass(kw_only=True)
class BERTEmbedder(TorchEmbedder):
    """
    Computes document embeddings with the BERT model. Takes a pretrained BERT model as
    input.
    """

    torch_model: BertModel


@dataclass(kw_only=True)
class LongformerEmbedder(TorchEmbedder):
    """
    Computes document embeddings with the Longformer model. Takes a pretrained Longformer model as
    input.
    """

    torch_model: LongformerModel
