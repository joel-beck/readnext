from pathlib import Path

import pytest

from readnext.modeling.language_models import (
    LanguageModelChoice,
    LanguageModelChoicePaths,
    get_cosine_similarities_path_from_choice,
    get_embeddings_path_from_choice,
    get_language_model_choice_paths,
)

language_model_choices = [
    LanguageModelChoice.TFIDF,
    LanguageModelChoice.BM25,
    LanguageModelChoice.WORD2VEC,
    LanguageModelChoice.GLOVE,
    LanguageModelChoice.FASTTEXT,
    LanguageModelChoice.BERT,
    LanguageModelChoice.SCIBERT,
    LanguageModelChoice.LONGFORMER,
]


def test_names_and_values() -> None:
    assert LanguageModelChoice.TFIDF.name == "TFIDF"
    assert LanguageModelChoice.TFIDF.value == "TFIDF"
    assert LanguageModelChoice.BM25.name == "BM25"
    assert LanguageModelChoice.BM25.value == "BM25"
    assert LanguageModelChoice.WORD2VEC.name == "WORD2VEC"
    assert LanguageModelChoice.WORD2VEC.value == "WORD2VEC"
    assert LanguageModelChoice.GLOVE.name == "GLOVE"
    assert LanguageModelChoice.GLOVE.value == "GLOVE"
    assert LanguageModelChoice.FASTTEXT.name == "FASTTEXT"
    assert LanguageModelChoice.FASTTEXT.value == "FASTTEXT"
    assert LanguageModelChoice.BERT.name == "BERT"
    assert LanguageModelChoice.BERT.value == "BERT"
    assert LanguageModelChoice.SCIBERT.name == "SCIBERT"
    assert LanguageModelChoice.SCIBERT.value == "SCIBERT"
    assert LanguageModelChoice.LONGFORMER.name == "LONGFORMER"
    assert LanguageModelChoice.LONGFORMER.value == "LONGFORMER"


def test_str_representation() -> None:
    assert str(LanguageModelChoice.TFIDF) == "TFIDF"
    assert str(LanguageModelChoice.BM25) == "BM25"
    assert str(LanguageModelChoice.WORD2VEC) == "WORD2VEC"
    assert str(LanguageModelChoice.GLOVE) == "GLOVE"
    assert str(LanguageModelChoice.FASTTEXT) == "FASTTEXT"
    assert str(LanguageModelChoice.BERT) == "BERT"
    assert str(LanguageModelChoice.SCIBERT) == "SCIBERT"
    assert str(LanguageModelChoice.LONGFORMER) == "LONGFORMER"


@pytest.mark.parametrize("choice", language_model_choices)
def test_language_model_choice_paths(choice: LanguageModelChoice) -> None:
    paths = get_language_model_choice_paths(choice)

    assert isinstance(paths, LanguageModelChoicePaths)
    assert isinstance(paths.embeddings, Path)
    assert isinstance(paths.cosine_similarities, Path)


@pytest.mark.parametrize("choice", language_model_choices)
def test_get_embeddings_path_from_choice(choice: LanguageModelChoice) -> None:
    embeddings_path = get_embeddings_path_from_choice(choice)
    assert isinstance(embeddings_path, Path)


@pytest.mark.parametrize("choice", language_model_choices)
def test_get_cosine_similarities_path_from_choice(choice: LanguageModelChoice) -> None:
    cosine_similarities_path = get_cosine_similarities_path_from_choice(choice)
    assert isinstance(cosine_similarities_path, Path)
