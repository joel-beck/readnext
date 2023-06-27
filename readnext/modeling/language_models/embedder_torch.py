from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import polars as pl
import torch

from readnext.modeling.language_models.embedder import AggregationStrategy
from readnext.utils.aliases import EmbeddingsFrame, TokenIds, TokenIdsFrame
from readnext.utils.protocols import BertModelProtocol, LongformerModelProtocol

TTorchModel = TypeVar("TTorchModel", bound=BertModelProtocol | LongformerModelProtocol)


@dataclass(kw_only=True)
class TorchEmbedder(ABC, Generic[TTorchModel]):
    """
    Abstract Base class for pytorch embedding models. All embedding models implement
    `compute_embedding_single_document()` and `compute_embeddings_frame()` methods.
    """

    token_ids_frame: TokenIdsFrame
    torch_model: TTorchModel
    device: torch.device
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
            return torch.mean(document_embeddings, dim=1)

        if aggregation_strategy.is_max:
            # Index 0 to only return the max values, not the indices
            return torch.max(document_embeddings, dim=1)[0]

        raise ValueError(f"Aggregation strategy `{aggregation_strategy}` is not implemented.")

    def compute_embeddings(self, tokenized_documents: list[TokenIds]) -> torch.Tensor:
        """
        Computes embeddings for multiple tokenized documents.

        Output is a Tensor with shape (n_documents_input, n_dimensions) with

        - n_documents_input: number of documents in provided input
        - n_dimensions: dimension of embedding space
        """

        bert_model = self.torch_model.to(self.device)  # type: ignore

        with torch.no_grad():
            bert_model.eval()

            # outputs is an ordered dictionary with keys `last_hidden_state` and `pooler_output`
            outputs = bert_model(torch.tensor(tokenized_documents, device=self.device))

        return outputs.last_hidden_state.detach().cpu()

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        """
        Computes embeddings for all tokenized abstracts in the documents frame.

        Output is a polars data frame with two columns named `d3_document_id` and
        `embedding`.
        """

        token_ids = self.token_ids_frame["token_ids"].to_list()

        outputs = self.compute_embeddings(token_ids)
        aggregated_outputs = self.aggregate_document_embeddings(outputs)

        return self.token_ids_frame.select("d3_document_id").with_columns(
            embedding=pl.Series(aggregated_outputs.tolist())
        )


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
