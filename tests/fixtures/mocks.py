import pytest

from readnext.utils.protocols import (
    BertModelProtocol,
    FastTextModelProtocol,
    LongformerModelProtocol,
    Word2VecModelProtocol,
)
from tests.mocks import (
    bert_model_mock,
    fasttext_model_mock,
    longformer_model_mock,
    word2vec_model_mock,
)


@pytest.fixture(scope="session")
def word2vec_model() -> Word2VecModelProtocol:
    return word2vec_model_mock()


@pytest.fixture(scope="session")
def fasttext_model() -> FastTextModelProtocol:
    return fasttext_model_mock()


@pytest.fixture(scope="session")
def bert_model() -> BertModelProtocol:
    return bert_model_mock()


@pytest.fixture(scope="session")
def longformer_model() -> LongformerModelProtocol:
    return longformer_model_mock()
