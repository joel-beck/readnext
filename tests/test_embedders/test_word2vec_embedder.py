import numpy as np

from readnext.modeling.language_models import Word2VecEmbedder
from readnext.utils import Tokens


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
