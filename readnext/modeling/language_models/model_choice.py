from enum import Enum


class LanguageModelChoice(Enum):
    tfidf = "tfidf"
    bm25 = "bm25"
    word2vec = "word2vec"
    glove = "glove"
    fasttext = "fasttext"
    bert = "bert"
    scibert = "scibert"
    longformer = "longformer"

    def __str__(self) -> str:
        return self.value
