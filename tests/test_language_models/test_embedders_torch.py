import numpy as np
import polars as pl
import pytest
import torch
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import (
    AggregationStrategy,
    BERTEmbedder,
    LongformerEmbedder,
    TorchEmbedder,
)
from readnext.utils import BertModelProtocol, LongformerModelProtocol, TokenIdsFrame

embedders = ["bert_embedder", "longformer_embedder"]


@pytest.mark.parametrize("embedder", lazy_fixture(embedders))
def test_aggregate_document_embeddings(embedder: TorchEmbedder) -> None:
    document_embeddings = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],  # shape (1, 2, 3)
        dtype=torch.float64,
    )

    assert torch.equal(
        embedder.aggregate_document_embeddings(document_embeddings, AggregationStrategy.mean),
        torch.tensor([[2.5, 3.5, 4.5]], dtype=torch.float64),  # mean along dimension 1
    )

    assert torch.equal(
        embedder.aggregate_document_embeddings(document_embeddings, AggregationStrategy.max),
        torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float64),  # max along dimension 1
    )


embedder_token_ids_frame_pairs = [
    ("bert_embedder", "bert_token_ids_frame"),
    ("longformer_embedder", "longformer_token_ids_frame"),
]


@pytest.mark.parametrize(
    ("embedder", "token_ids_frame"),
    [(lazy_fixture(x), lazy_fixture(y)) for x, y in embedder_token_ids_frame_pairs],
)
def test_compute_embedding_single_document(
    embedder: TorchEmbedder, token_ids_frame: TokenIdsFrame
) -> None:
    first_document_token_ids = token_ids_frame["token_ids"][0]
    embeddings_single_document = embedder.compute_embedding_single_document(
        first_document_token_ids
    )

    assert isinstance(embeddings_single_document, list)
    assert all(
        isinstance(embedding_dimension, float) for embedding_dimension in embeddings_single_document
    )
    assert all(len())


@pytest.mark.parametrize("embedder", lazy_fixture(embedders))
def test_compute_embeddings_frame(embedder: TorchEmbedder) -> None:
    embeddings_frame = embedder.compute_embeddings_frame()

    assert isinstance(embeddings_frame, pl.DataFrame)
    assert all(isinstance(key, int) for key in embeddings_frame)
    assert all(isinstance(value, np.ndarray) for value in embeddings_frame.values())

    assert len(embeddings_frame) == 3
    assert all(len(value) == 768 for value in embeddings_frame.values())


def test_kw_only_initialization_bert_embedder(bert_model: BertModelProtocol) -> None:
    with pytest.raises(TypeError):
        BERTEmbedder(
            {-1: torch.Tensor([1, 2, 3])},  # type: ignore
            bert_model,
        )


def test_kw_only_initialization_longformer_embedder(
    longformer_model: LongformerModelProtocol,
) -> None:
    with pytest.raises(TypeError):
        LongformerEmbedder(
            {-1: torch.Tensor([1, 2, 3])},  # type: ignore
            longformer_model,
        )
