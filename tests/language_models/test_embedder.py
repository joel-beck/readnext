import numpy as np
import polars as pl
import pytest
import torch
from numpy.testing import assert_almost_equal, assert_array_equal
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import (
    AggregationStrategy,
    BERTEmbedder,
    BM25Embedder,
    GensimEmbedder,
    LongformerEmbedder,
    TFIDFEmbedder,
    TorchEmbedder,
    tfidf,
)
from readnext.utils.aliases import TokenIds, Tokens
from readnext.utils.protocols import BertModelProtocol, LongformerModelProtocol

keyword_based_embedder_fixtures = [lazy_fixture("tfidf_embedder"), lazy_fixture("bm25_embedder")]
gensim_embedder_fixtures = [lazy_fixture("word2vec_embedder"), lazy_fixture("fasttext_embedder")]
torch_embedder_fixtures = [lazy_fixture("bert_embedder"), lazy_fixture("longformer_embedder")]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("embedder", gensim_embedder_fixtures)
def test_word_embeddings_to_document_embedding(embedder: GensimEmbedder) -> None:
    # `word_embeddings_to_document_embedding()` takes numpy arrays as input, conversion
    # to lists occurs afterwards
    word_embeddings_per_document = np.array([[1, 2, 3], [4, 5, 6]])

    embedder.aggregation_strategy = AggregationStrategy.mean

    assert_almost_equal(
        embedder.token_embeddings_to_document_embedding(word_embeddings_per_document),
        np.array([2.5, 3.5, 4.5]),
    )

    embedder.aggregation_strategy = AggregationStrategy.max

    assert_array_equal(
        embedder.token_embeddings_to_document_embedding(word_embeddings_per_document),
        np.array([4, 5, 6]),
    )


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("embedder", torch_embedder_fixtures)
def test_aggregate_document_embeddings(embedder: TorchEmbedder) -> None:
    document_embeddings = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],  # shape (1, 2, 3)
        dtype=torch.float64,
    )

    embedder.aggregation_strategy = AggregationStrategy.mean

    assert torch.equal(
        embedder.aggregate_document_embeddings(document_embeddings),
        torch.tensor([[2.5, 3.5, 4.5]], dtype=torch.float64),  # mean along dimension 1
    )

    embedder.aggregation_strategy = AggregationStrategy.max

    assert torch.equal(
        embedder.aggregate_document_embeddings(document_embeddings),
        torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float64),  # max along dimension 1
    )


@pytest.mark.parametrize(
    "embedder",
    [
        *[pytest.param(fixture) for fixture in keyword_based_embedder_fixtures],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in gensim_embedder_fixtures + torch_embedder_fixtures
        ],
    ],
)
def test_compute_embeddings_frame(embedder: GensimEmbedder | TorchEmbedder) -> None:
    embeddings_frame = embedder.compute_embeddings_frame()

    assert isinstance(embeddings_frame, pl.DataFrame)
    assert embeddings_frame.width == 2
    assert embeddings_frame.columns == ["d3_document_id", "embedding"]
    assert embeddings_frame.dtypes == [pl.Int64, pl.List(pl.Float64)]


def test_embedding_dimension_tfidf(tfidf_embedder: TFIDFEmbedder, toy_tokens: Tokens) -> None:
    single_embedding = tfidf_embedder.compute_embedding_single_document(toy_tokens)

    embeddings_frame = tfidf_embedder.compute_embeddings_frame()
    corpus_vocabulary_size = tfidf_embedder.tokens_frame["tokens"].explode().n_unique()

    assert len(single_embedding) == corpus_vocabulary_size
    assert (embeddings_frame["embedding"].list.lengths() == corpus_vocabulary_size).all()


def test_embedding_dimension_bm25(bm25_embedder: BM25Embedder, toy_tokens: Tokens) -> None:
    single_embedding = bm25_embedder.compute_embedding_single_document(toy_tokens)

    embeddings_frame = bm25_embedder.compute_embeddings_frame()
    corpus_vocabulary_size = bm25_embedder.tokens_frame["tokens"].explode().n_unique()

    assert len(single_embedding) == corpus_vocabulary_size
    assert (embeddings_frame["embedding"].list.lengths() == corpus_vocabulary_size).all()


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("embedder", gensim_embedder_fixtures)
def test_embedding_dimension_gensim(embedder: GensimEmbedder, toy_tokens: Tokens) -> None:
    single_embedding = embedder.compute_embedding_single_document(toy_tokens)
    embeddings_frame = embedder.compute_embeddings_frame()

    assert len(single_embedding) == 300
    assert (embeddings_frame["embedding"].list.lengths() == 300).all()


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("embedder", torch_embedder_fixtures)
def test_embedding_dimension_torch(embedder: TorchEmbedder, toy_token_ids: TokenIds) -> None:
    single_embedding = embedder.compute_embedding_single_document(toy_token_ids)
    embeddings_frame = embedder.compute_embeddings_frame()

    assert len(single_embedding) == 768
    assert (embeddings_frame["embedding"].list.lengths() == 768).all()


def test_kw_only_initialization_tfidf_embedder() -> None:
    with pytest.raises(TypeError):
        TFIDFEmbedder(pl.DataFrame(), tfidf)  # type: ignore


def test_kw_only_initialization_bert_embedder(bert_model: BertModelProtocol) -> None:
    with pytest.raises(TypeError):
        BERTEmbedder(
            pl.DataFrame(),  # type: ignore
            bert_model,
        )


def test_kw_only_initialization_longformer_embedder(
    longformer_model: LongformerModelProtocol,
) -> None:
    with pytest.raises(TypeError):
        LongformerEmbedder(
            pl.DataFrame(),  # type: ignore
            longformer_model,
        )
