from readnext.modeling.language_models import LanguageModelChoice


def test_names_and_values() -> None:
    assert LanguageModelChoice.tfidf.name == "tfidf"
    assert LanguageModelChoice.tfidf.value == "tfidf"
    assert LanguageModelChoice.bm25.name == "bm25"
    assert LanguageModelChoice.bm25.value == "bm25"
    assert LanguageModelChoice.word2vec.name == "word2vec"
    assert LanguageModelChoice.word2vec.value == "word2vec"
    assert LanguageModelChoice.glove.name == "glove"
    assert LanguageModelChoice.glove.value == "glove"
    assert LanguageModelChoice.fasttext.name == "fasttext"
    assert LanguageModelChoice.fasttext.value == "fasttext"
    assert LanguageModelChoice.bert.name == "bert"
    assert LanguageModelChoice.bert.value == "bert"
    assert LanguageModelChoice.scibert.name == "scibert"
    assert LanguageModelChoice.scibert.value == "scibert"
    assert LanguageModelChoice.longformer.name == "longformer"
    assert LanguageModelChoice.longformer.value == "longformer"


def test_str_representation() -> None:
    assert str(LanguageModelChoice.tfidf) == "tfidf"
    assert str(LanguageModelChoice.bm25) == "bm25"
    assert str(LanguageModelChoice.word2vec) == "word2vec"
    assert str(LanguageModelChoice.glove) == "glove"
    assert str(LanguageModelChoice.fasttext) == "fasttext"
    assert str(LanguageModelChoice.bert) == "bert"
    assert str(LanguageModelChoice.scibert) == "scibert"
    assert str(LanguageModelChoice.longformer) == "longformer"
