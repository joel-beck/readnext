import numpy as np
import pytest

from readnext.modeling.language_models import TFIDFEmbedder, tfidf
from readnext.utils import Tokens, TokensMapping


@pytest.fixture(scope="module")
def num_unique_corpus_tokens(spacy_tokenized_abstracts: list[Tokens]) -> int:
    # vocabulary has 18 unique tokens
    unique_corpus_tokens = {token for tokens in spacy_tokenized_abstracts for token in tokens}
    return len(unique_corpus_tokens)


@pytest.fixture(scope="module")
def tfidf_embedder(spacy_tokens_mapping: TokensMapping) -> TFIDFEmbedder:
    return TFIDFEmbedder(keyword_algorithm=tfidf, tokens_mapping=spacy_tokens_mapping)


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


def test_compute_embeddings_mapping(
    tfidf_embedder: TFIDFEmbedder,
    spacy_tokenized_abstracts: list[Tokens],
    num_unique_corpus_tokens: int,
) -> None:
    tfidf_embeddings_mapping = tfidf_embedder.compute_embeddings_mapping()

    assert isinstance(tfidf_embeddings_mapping, dict)
    assert all(isinstance(key, int) for key in tfidf_embeddings_mapping)
    assert all(isinstance(value, np.ndarray) for value in tfidf_embeddings_mapping.values())

    assert len(tfidf_embeddings_mapping) == len(spacy_tokenized_abstracts)
    assert all(
        len(value) == num_unique_corpus_tokens for value in tfidf_embeddings_mapping.values()
    )
