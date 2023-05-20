import numpy as np
import pytest
import torch
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import AggregationStrategy, TorchEmbedder
from readnext.utils.aliases import TokensIdMapping

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


embedder_tokens_id_mapping_pairs = [
    ("bert_embedder", "bert_tokens_id_mapping"),
    ("longformer_embedder", "longformer_tokens_id_mapping"),
]


@pytest.mark.parametrize(
    ("embedder", "tokens_id_mapping"),
    [(lazy_fixture(x), lazy_fixture(y)) for x, y in embedder_tokens_id_mapping_pairs],
)
def test_compute_embedding_single_document(
    embedder: TorchEmbedder, tokens_id_mapping: TokensIdMapping
) -> None:
    first_document_token_ids = tokens_id_mapping[1]
    embeddings_single_document = embedder.compute_embedding_single_document(
        first_document_token_ids
    )

    assert isinstance(embeddings_single_document, np.ndarray)
    assert embeddings_single_document.dtype == np.dtype("float32")
    assert embeddings_single_document.shape == (768,)


@pytest.mark.parametrize("embedder", lazy_fixture(embedders))
def test_compute_embeddings_mapping(embedder: TorchEmbedder) -> None:
    embeddings_mapping = embedder.compute_embeddings_mapping()

    assert isinstance(embeddings_mapping, dict)
    assert all(isinstance(key, int) for key in embeddings_mapping)
    assert all(isinstance(value, np.ndarray) for value in embeddings_mapping.values())

    assert len(embeddings_mapping) == 3
    assert all(len(value) == 768 for value in embeddings_mapping.values())
