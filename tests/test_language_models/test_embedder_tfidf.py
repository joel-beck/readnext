import numpy as np
import pytest

from readnext.modeling.language_models import TFIDFEmbedder, tfidf
from readnext.utils import Tokens


def test_compute_embedding_single_document(
    tfidf_embedder: TFIDFEmbedder,
    spacy_tokenized_abstracts: list[Tokens],
    num_unique_corpus_tokens: int,
) -> None:
    tfidf_embeddings_single_document = tfidf_embedder.compute_embedding_single_document(
        spacy_tokenized_abstracts[0]
    )
    assert isinstance(tfidf_embeddings_single_document, np.ndarray)
    assert tfidf_embeddings_single_document.dtype == np.float64
    assert tfidf_embeddings_single_document.shape == (num_unique_corpus_tokens,)


def test_compute_embeddings_frame(
    tfidf_embedder: TFIDFEmbedder,
    spacy_tokenized_abstracts: list[Tokens],
    num_unique_corpus_tokens: int,
) -> None:
    tfidf_embeddings_frame = tfidf_embedder.compute_embeddings_frame()

    assert isinstance(tfidf_embeddings_frame, dict)
    assert all(isinstance(key, int) for key in tfidf_embeddings_frame)
    assert all(isinstance(value, np.ndarray) for value in tfidf_embeddings_frame.values())

    assert len(tfidf_embeddings_frame) == len(spacy_tokenized_abstracts)
    assert all(len(value) == num_unique_corpus_tokens for value in tfidf_embeddings_frame.values())


def test_kw_only_initialization_tfidf_embedder() -> None:
    with pytest.raises(TypeError):
        TFIDFEmbedder({-1: ["a", "b", "c"]}, tfidf)  # type: ignore
