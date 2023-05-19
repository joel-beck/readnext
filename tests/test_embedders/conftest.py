import pytest

from readnext.modeling.language_models import (
    FastTextEmbedder,
    TFIDFEmbedder,
    Word2VecEmbedder,
    tfidf,
)
from readnext.utils import FastTextModelProtocol, Tokens, TokensMapping, Word2VecModelProtocol


@pytest.fixture(scope="session")
def num_unique_corpus_tokens(spacy_tokenized_abstracts: list[Tokens]) -> int:
    # vocabulary has 18 unique tokens
    unique_corpus_tokens = {token for tokens in spacy_tokenized_abstracts for token in tokens}
    return len(unique_corpus_tokens)


@pytest.fixture(scope="session")
def tfidf_embedder(spacy_tokens_mapping: TokensMapping) -> TFIDFEmbedder:
    return TFIDFEmbedder(keyword_algorithm=tfidf, tokens_mapping=spacy_tokens_mapping)


@pytest.fixture(scope="session")
def word2vec_embedder(
    spacy_tokens_mapping: TokensMapping, word2vec_model: Word2VecModelProtocol
) -> Word2VecEmbedder:
    return Word2VecEmbedder(spacy_tokens_mapping, word2vec_model)


@pytest.fixture(scope="session")
def fasttext_embedder(
    spacy_tokens_mapping: TokensMapping, fasttext_model: FastTextModelProtocol
) -> FastTextEmbedder:
    return FastTextEmbedder(spacy_tokens_mapping, fasttext_model)
