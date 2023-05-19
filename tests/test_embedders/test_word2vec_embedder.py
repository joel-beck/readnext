import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from readnext.modeling.language_models import AggregationStrategy, Word2VecEmbedder
from readnext.utils import Tokens, TokensMapping, Word2VecModelProtocol


@pytest.fixture(scope="module")
def num_unique_corpus_tokens(spacy_tokenized_abstracts: list[Tokens]) -> int:
    # vocabulary has 18 unique tokens
    unique_corpus_tokens = {token for tokens in spacy_tokenized_abstracts for token in tokens}
    return len(unique_corpus_tokens)


@pytest.fixture(scope="module")
def word2vec_embedder(
    spacy_tokens_mapping: TokensMapping, word2vec_model: Word2VecModelProtocol
) -> Word2VecEmbedder:
    return Word2VecEmbedder(spacy_tokens_mapping, word2vec_model)


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


def test_compute_embedding_single_document(
    word2vec_embedder: Word2VecEmbedder, spacy_tokenized_abstracts: list[Tokens]
) -> None:
    word2vec_embeddings_single_document = word2vec_embedder.compute_embedding_single_document(
        spacy_tokenized_abstracts[0]
    )
    assert isinstance(word2vec_embeddings_single_document, np.ndarray)
    assert word2vec_embeddings_single_document.dtype == np.float64
    assert word2vec_embeddings_single_document.shape == (300,)


def test_compute_embeddings_mapping(
    word2vec_embedder: Word2VecEmbedder,
    spacy_tokenized_abstracts: list[Tokens],
) -> None:
    word2vec_embeddings_mapping = word2vec_embedder.compute_embeddings_mapping()

    assert isinstance(word2vec_embeddings_mapping, dict)
    assert all(isinstance(key, int) for key in word2vec_embeddings_mapping)
    assert all(isinstance(value, np.ndarray) for value in word2vec_embeddings_mapping.values())

    assert len(word2vec_embeddings_mapping) == len(spacy_tokenized_abstracts)
    assert all(len(value) == 300 for value in word2vec_embeddings_mapping.values())
