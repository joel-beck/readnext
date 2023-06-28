"""
Tests for loading correct language models. All used fixtures in this file use the
`load_language_model` function which is the object of interest here.
"""

import pytest
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, LongformerModel


def test_tfidf_model(tfidf_model: TfidfVectorizer) -> None:
    assert isinstance(tfidf_model, TfidfVectorizer)


@pytest.mark.slow
@pytest.mark.skip_ci
def test_word2vec_model(word2vec_model: KeyedVectors) -> None:
    assert isinstance(word2vec_model, KeyedVectors)


@pytest.mark.slow
@pytest.mark.skip_ci
def test_glove_model(glove_model: KeyedVectors) -> None:
    assert isinstance(glove_model, KeyedVectors)


@pytest.mark.slow
@pytest.mark.skip_ci
def fasttext_model(fasttext_model: FastText) -> None:
    assert isinstance(fasttext_model, FastText)


@pytest.mark.slow
@pytest.mark.skip_ci
def bert_model(bert_model: BertModel) -> None:
    assert isinstance(bert_model, BertModel)


@pytest.mark.slow
@pytest.mark.skip_ci
def scibert_model(scibert_model: BertModel) -> None:
    assert isinstance(scibert_model, BertModel)


@pytest.mark.slow
@pytest.mark.skip_ci
def longformer_model(longformer_model: LongformerModel) -> None:
    assert isinstance(longformer_model, LongformerModel)
