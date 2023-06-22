import numpy as np
import polars as pl
import pytest
import torch
from numpy.testing import assert_almost_equal, assert_array_equal
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import (
    AggregationStrategy,
    BERTEmbedder,
    GensimEmbedder,
    LongformerEmbedder,
    TFIDFEmbedder,
    TorchEmbedder,
    tfidf,
)
from readnext.utils.protocols import BertModelProtocol, LongformerModelProtocol

gensim_embedder_fixtures = [lazy_fixture("word2vec_embedder"), lazy_fixture("fasttext_embedder")]

tfidf_gensim_embedder_fixtures = [lazy_fixture("tfidf_embedder"), *gensim_embedder_fixtures]

torch_embedder_fixtures = [lazy_fixture("bert_embedder"), lazy_fixture("longformer_embedder")]

embedder_fixtures = tfidf_gensim_embedder_fixtures + torch_embedder_fixtures


@pytest.mark.parametrize("embedder", tfidf_gensim_embedder_fixtures)
def test_word_embeddings_to_document_embedding(embedder: GensimEmbedder) -> None:
    # `word_embeddings_to_document_embedding()` takes numpy arrays as input, conversion
    # to lists occurs afterwards
    word_embeddings_per_document = np.array([[1, 2, 3], [4, 5, 6]])

    assert_almost_equal(
        embedder.word_embeddings_to_document_embedding(
            word_embeddings_per_document, AggregationStrategy.mean
        ),
        np.array([2.5, 3.5, 4.5]),
    )

    assert_array_equal(
        embedder.word_embeddings_to_document_embedding(
            word_embeddings_per_document, AggregationStrategy.max
        ),
        np.array([4, 5, 6]),
    )


@pytest.mark.parametrize("embedder", torch_embedder_fixtures)
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


@pytest.mark.parametrize("embedder", embedder_fixtures)
def test_compute_embeddings_frame(embedder: GensimEmbedder | TorchEmbedder) -> None:
    embeddings_frame = embedder.compute_embeddings_frame()

    assert isinstance(embeddings_frame, pl.DataFrame)
    assert embeddings_frame.shape[1] == 2
    assert embeddings_frame.columns == ["d3_document_id", "embedding"]
    assert embeddings_frame.dtypes == [pl.Int64, pl.List(pl.Float64)]


def test_embedding_dimension_tfidf(tfidf_embedder: TFIDFEmbedder) -> None:
    embeddings_frame = tfidf_embedder.compute_embeddings_frame()
    corpus_vocabulary_size = tfidf_embedder.tokens_frame["tokens"].explode().n_unique()

    assert (embeddings_frame["embedding"].list.lengths() == corpus_vocabulary_size).all()


@pytest.mark.parametrize("embedder", gensim_embedder_fixtures)
def test_embedding_dimension_gensim(embedder: GensimEmbedder) -> None:
    embeddings_frame = embedder.compute_embeddings_frame()
    assert (embeddings_frame["embedding"].list.lengths() == 300).all()


@pytest.mark.parametrize("embedder", torch_embedder_fixtures)
def test_embedding_dimension_torch(embedder: GensimEmbedder) -> None:
    embeddings_frame = embedder.compute_embeddings_frame()
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
