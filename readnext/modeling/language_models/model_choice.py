from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from readnext.config import ResultsPaths
from readnext.utils import EmbeddingsFrame, ScoresFrame, read_df_from_parquet, status_update


class LanguageModelChoice(Enum):
    tfidf = "TF-IDF"
    bm25 = "BM25"
    word2vec = "Word2Vec"
    glove = "GloVe"
    fasttext = "FastText"
    bert = "BERT"
    scibert = "SciBERT"
    longformer = "Longformer"

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
                embeddings=ResultsPaths.language_models.tfidf_embeddings_parquet,
                cosine_similarities=ResultsPaths.language_models.tfidf_cosine_similarities_parquet,
            )
        case LanguageModelChoice.bm25:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.bm25_embeddings_parquet,
                cosine_similarities=ResultsPaths.language_models.bm25_cosine_similarities_parquet,
            )
        case LanguageModelChoice.word2vec:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.word2vec_embeddings_parquet,
                cosine_similarities=ResultsPaths.language_models.word2vec_cosine_similarities_parquet,
            )
        case LanguageModelChoice.glove:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.glove_embeddings_parquet,
                cosine_similarities=ResultsPaths.language_models.glove_cosine_similarities_parquet,
            )
        case LanguageModelChoice.fasttext:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.fasttext_embeddings_parquet,
                cosine_similarities=ResultsPaths.language_models.fasttext_cosine_similarities_parquet,
            )
        case LanguageModelChoice.bert:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.bert_embeddings_parquet,
                cosine_similarities=ResultsPaths.language_models.bert_cosine_similarities_parquet,
            )
        case LanguageModelChoice.scibert:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.scibert_embeddings_parquet,
                cosine_similarities=ResultsPaths.language_models.scibert_cosine_similarities_parquet,
            )
        case LanguageModelChoice.longformer:
            return LanguageModelChoicePaths(
                embeddings=ResultsPaths.language_models.longformer_embeddings_parquet,
                cosine_similarities=ResultsPaths.language_models.longformer_cosine_similarities_parquet,
            )
        case _:
            raise ValueError(f"Invalid language model choice: {language_model_choice}")


def get_cosine_similarities_path_from_choice(language_model_choice: LanguageModelChoice) -> Path:
    return get_language_model_choice_paths(language_model_choice).cosine_similarities


def get_embeddings_path_from_choice(language_model_choice: LanguageModelChoice) -> Path:
    return get_language_model_choice_paths(language_model_choice).embeddings


def load_cosine_similarities_from_choice(language_model_choice: LanguageModelChoice) -> ScoresFrame:
    cosine_similarities_path = get_cosine_similarities_path_from_choice(language_model_choice)
    return read_df_from_parquet(cosine_similarities_path)


@status_update("Loading pretrained embeddings")
def load_embeddings_from_choice(language_model_choice: LanguageModelChoice) -> EmbeddingsFrame:
    embeddings_path = get_embeddings_path_from_choice(language_model_choice)
    return read_df_from_parquet(embeddings_path)
