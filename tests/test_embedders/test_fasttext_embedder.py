import numpy as np
import pytest

from readnext.modeling.language_models import FastTextEmbedder
from readnext.utils import FastTextModelProtocol, Tokens, TokensMapping


@pytest.fixture(scope="module")
def fasttext_embedder(
    spacy_tokens_mapping: TokensMapping, fasttext_model: FastTextModelProtocol
) -> FastTextEmbedder:
    return FastTextEmbedder(spacy_tokens_mapping, fasttext_model)


def test_compute_embedding_single_document(
    fasttext_embedder: FastTextEmbedder, spacy_tokenized_abstracts: list[Tokens]
) -> None:
    fasttext_embeddings_single_document = fasttext_embedder.compute_embedding_single_document(
        spacy_tokenized_abstracts[0]
    )
    assert isinstance(fasttext_embeddings_single_document, np.ndarray)
    assert fasttext_embeddings_single_document.dtype == np.float64
    assert fasttext_embeddings_single_document.shape == (300,)


def test_compute_embeddings_mapping(
    fasttext_embedder: FastTextEmbedder,
    spacy_tokenized_abstracts: list[Tokens],
) -> None:
    word2vec_embeddings_mapping = fasttext_embedder.compute_embeddings_mapping()

    assert isinstance(word2vec_embeddings_mapping, dict)
    assert all(isinstance(key, int) for key in word2vec_embeddings_mapping)
    assert all(isinstance(value, np.ndarray) for value in word2vec_embeddings_mapping.values())

    assert len(word2vec_embeddings_mapping) == len(spacy_tokenized_abstracts)
    assert all(len(value) == 300 for value in word2vec_embeddings_mapping.values())
