import pytest

from readnext.modeling.language_models import Word2VecEmbedder
from readnext.utils import TokensMapping, Word2VecModelProtocol


@pytest.fixture(scope="module")
def word2vec_embedder(
    spacy_tokens_mapping: TokensMapping, word2vec_model: Word2VecModelProtocol
) -> Word2VecEmbedder:
    return Word2VecEmbedder(spacy_tokens_mapping, word2vec_model)
