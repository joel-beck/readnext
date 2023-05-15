from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from readnext.config import ResultsPaths
from readnext.utils import load_df_from_pickle


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


@dataclass
class LanguageModelChoicePaths:
    embeddings: Path
    cosine_similarities: Path


def get_language_model_choice_paths(
    language_model_choice: LanguageModelChoice,
) -> LanguageModelChoicePaths:
    match language_model_choice:
        case LanguageModelChoice.tfidf:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.tfidf_embeddings_mapping_most_cited_pkl,
                cosine_similarities=ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl,
            )
        case LanguageModelChoice.bm25:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.bm25_embeddings_mapping_most_cited_pkl,
                cosine_similarities=ResultsPaths.language_models.bm25_cosine_similarities_most_cited_pkl,
            )
        case LanguageModelChoice.word2vec:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.word2vec_embeddings_mapping_most_cited_pkl,
                cosine_similarities=ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl,
            )
        case LanguageModelChoice.glove:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.glove_embeddings_mapping_most_cited_pkl,
                cosine_similarities=ResultsPaths.language_models.glove_cosine_similarities_most_cited_pkl,
            )
        case LanguageModelChoice.fasttext:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.fasttext_embeddings_mapping_most_cited_pkl,
                cosine_similarities=ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl,
            )
        case LanguageModelChoice.bert:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.bert_embeddings_mapping_most_cited_pkl,
                cosine_similarities=ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl,
            )
        case LanguageModelChoice.scibert:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.scibert_embeddings_mapping_most_cited_pkl,
                cosine_similarities=ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl,
            )
        case LanguageModelChoice.longformer:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.longformer_embeddings_mapping_most_cited_pkl,
                cosine_similarities=ResultsPaths.language_models.longformer_cosine_similarities_most_cited_pkl,
            )
        case _:
            raise ValueError(f"Invalid language model choice: {language_model_choice}")


def get_cosine_similarities_path_from_choice(language_model_choice: LanguageModelChoice) -> Path:
    return get_language_model_choice_paths(language_model_choice).cosine_similarities


def get_embeddings_path_from_choice(language_model_choice: LanguageModelChoice) -> Path:
    return get_language_model_choice_paths(language_model_choice).embeddings


def load_cosine_similarities_from_choice(
    language_model_choice: LanguageModelChoice,
) -> pd.DataFrame:
    cosine_similarities_path = get_cosine_similarities_path_from_choice(language_model_choice)
    return load_df_from_pickle(cosine_similarities_path)


def load_embeddings_from_choice(
    language_model_choice: LanguageModelChoice,
) -> pd.DataFrame:
    embeddings_mapping_path = get_embeddings_path_from_choice(language_model_choice)
    return load_df_from_pickle(embeddings_mapping_path)
