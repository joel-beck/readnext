import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, LongformerModel

from readnext.config import ModelVersions
from readnext.utils.protocols import FastTextModelProtocol, Word2VecModelProtocol
from tests.mocks import fasttext_model_mock, word2vec_model_mock


@pytest.fixture(scope="session")
def tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer()


@pytest.fixture(scope="session")
def word2vec_model() -> Word2VecModelProtocol:
    return word2vec_model_mock()


@pytest.fixture(scope="session")
def fasttext_model() -> FastTextModelProtocol:
    return fasttext_model_mock()


@pytest.fixture(scope="session")
def bert_model() -> BertModel:
    return BertModel.from_pretrained(ModelVersions.bert)


@pytest.fixture(scope="session")
def longformer_model() -> LongformerModel:
    return LongformerModel.from_pretrained(ModelVersions.longformer)
