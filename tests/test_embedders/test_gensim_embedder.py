import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from readnext.modeling.language_models import AggregationStrategy, Word2VecEmbedder


def test_word_embeddings_to_document_embedding(word2vec_embedder: Word2VecEmbedder) -> None:
    word_embeddings_per_document = np.array([[1, 2, 3], [4, 5, 6]])

    assert_almost_equal(
        word2vec_embedder.word_embeddings_to_document_embedding(
            word_embeddings_per_document, AggregationStrategy.mean
        ),
        np.array([2.5, 3.5, 4.5]),
    )

    assert_array_equal(
        word2vec_embedder.word_embeddings_to_document_embedding(
            word_embeddings_per_document, AggregationStrategy.max
        ),
        np.array([4, 5, 6]),
    )
