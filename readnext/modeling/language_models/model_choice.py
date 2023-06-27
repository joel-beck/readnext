from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from readnext.config import ResultsPaths


class LanguageModelChoice(Enum):
    TFIDF = "TFIDF"
    BM25 = "BM25"
    WORD2VEC = "WORD2VEC"
    GLOVE = "GLOVE"
    FASTTEXT = "FASTTEXT"
    BERT = "BERT"
    SCIBERT = "SCIBERT"
    LONGFORMER = "LONGFORMER"

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
        case LanguageModelChoice.TFIDF:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.tfidf_embeddings_frame_parquet,
                cosine_similarities=ResultsPaths.language_models.tfidf_cosine_similarities_parquet,
            )
        case LanguageModelChoice.BM25:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.bm25_embeddings_frame_parquet,
                cosine_similarities=ResultsPaths.language_models.bm25_cosine_similarities_parquet,
            )
        case LanguageModelChoice.WORD2VEC:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.word2vec_embeddings_frame_parquet,
                cosine_similarities=ResultsPaths.language_models.word2vec_cosine_similarities_parquet,
            )
        case LanguageModelChoice.GLOVE:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.glove_embeddings_frame_parquet,
                cosine_similarities=ResultsPaths.language_models.glove_cosine_similarities_parquet,
            )
        case LanguageModelChoice.FASTTEXT:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.fasttext_embeddings_frame_parquet,
                cosine_similarities=ResultsPaths.language_models.fasttext_cosine_similarities_parquet,
            )
        case LanguageModelChoice.BERT:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.bert_embeddings_frame_parquet,
                cosine_similarities=ResultsPaths.language_models.bert_cosine_similarities_parquet,
            )
        case LanguageModelChoice.SCIBERT:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.scibert_embeddings_frame_parquet,
                cosine_similarities=ResultsPaths.language_models.scibert_cosine_similarities_parquet,
            )
        case LanguageModelChoice.LONGFORMER:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.longformer_embeddings_frame_parquet,
                cosine_similarities=ResultsPaths.language_models.longformer_cosine_similarities_parquet,
            )
        case _:
            raise ValueError(f"Invalid language model choice: {language_model_choice}")


def get_embeddings_path_from_choice(language_model_choice: LanguageModelChoice) -> Path:
    return get_language_model_choice_paths(language_model_choice).embeddings


def get_cosine_similarities_path_from_choice(language_model_choice: LanguageModelChoice) -> Path:
    return get_language_model_choice_paths(language_model_choice).cosine_similarities
