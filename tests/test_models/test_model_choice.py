from readnext.modeling.language_models import LanguageModelChoice


def test_names_and_values() -> None:
    assert LanguageModelChoice.tfidf.name == "tfidf"
    assert LanguageModelChoice.tfidf.value == "TF-IDF"
    assert LanguageModelChoice.bm25.name == "bm25"
    assert LanguageModelChoice.bm25.value == "BM25"
    assert LanguageModelChoice.word2vec.name == "word2vec"
    assert LanguageModelChoice.word2vec.value == "Word2Vec"
    assert LanguageModelChoice.glove.name == "glove"
    assert LanguageModelChoice.glove.value == "GloVe"
    assert LanguageModelChoice.fasttext.name == "fasttext"
    assert LanguageModelChoice.fasttext.value == "FastText"
    assert LanguageModelChoice.bert.name == "bert"
    assert LanguageModelChoice.bert.value == "BERT"
    assert LanguageModelChoice.scibert.name == "scibert"
    assert LanguageModelChoice.scibert.value == "SciBERT"
    assert LanguageModelChoice.longformer.name == "longformer"
    assert LanguageModelChoice.longformer.value == "Longformer"


def test_str_representation() -> None:
    assert str(LanguageModelChoice.tfidf) == "TF-IDF"
    assert str(LanguageModelChoice.bm25) == "BM25"
    assert str(LanguageModelChoice.word2vec) == "Word2Vec"
    assert str(LanguageModelChoice.glove) == "GloVe"
    assert str(LanguageModelChoice.fasttext) == "FastText"
    assert str(LanguageModelChoice.bert) == "BERT"
    assert str(LanguageModelChoice.scibert) == "SciBERT"
    assert str(LanguageModelChoice.longformer) == "Longformer"
