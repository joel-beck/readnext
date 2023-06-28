import pytest
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, LongformerModel

from readnext.modeling.language_models import LanguageModelChoice, load_language_model


@pytest.fixture(scope="session")
def tfidf_model() -> TfidfVectorizer:
    return load_language_model(LanguageModelChoice.TFIDF)


@pytest.fixture(scope="session")
def word2vec_model() -> KeyedVectors:
    return load_language_model(LanguageModelChoice.WORD2VEC)


@pytest.fixture(scope="session")
def glove_model() -> KeyedVectors:
    return load_language_model(LanguageModelChoice.GLOVE)


@pytest.fixture(scope="session")
def fasttext_model() -> FastText:
    return load_language_model(LanguageModelChoice.FASTTEXT)


@pytest.fixture(scope="session")
def bert_model() -> BertModel:
    return load_language_model(LanguageModelChoice.BERT)


@pytest.fixture(scope="session")
def scibert_model() -> BertModel:
    return load_language_model(LanguageModelChoice.SCIBERT)


@pytest.fixture(scope="session")
def longformer_model() -> LongformerModel:
    return load_language_model(LanguageModelChoice.LONGFORMER)
