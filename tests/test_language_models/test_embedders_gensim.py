import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import AggregationStrategy, GensimEmbedder
from readnext.utils import Tokens

embedders = ["word2vec_embedder", "fasttext_embedder"]


@pytest.mark.parametrize("embedder", lazy_fixture(embedders))
def test_word_embeddings_to_document_embedding(embedder: GensimEmbedder) -> None:
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


@pytest.mark.parametrize("embedder", lazy_fixture(embedders))
def test_compute_embedding_single_document(
    embedder: GensimEmbedder, spacy_tokenized_abstracts: list[Tokens]
) -> None:
    embeddings_single_document = embedder.compute_embedding_single_document(
        spacy_tokenized_abstracts[0]
    )
    assert isinstance(embeddings_single_document, np.ndarray)
    assert embeddings_single_document.dtype == np.float64
    assert embeddings_single_document.shape == (300,)


@pytest.mark.parametrize("embedder", lazy_fixture(embedders))
def test_compute_embeddings_frame(
    embedder: GensimEmbedder, spacy_tokenized_abstracts: list[Tokens]
) -> None:
    embeddings_frame = embedder.compute_embeddings_frame()

    assert isinstance(embeddings_frame, dict)
    assert all(isinstance(key, int) for key in embeddings_frame)
    assert all(isinstance(value, np.ndarray) for value in embeddings_frame.values())

    assert len(embeddings_frame) == len(spacy_tokenized_abstracts)
    assert all(len(value) == 300 for value in embeddings_frame.values())
