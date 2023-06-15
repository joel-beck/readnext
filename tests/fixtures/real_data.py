import pytest

from readnext.config import DataPaths, ResultsPaths
from readnext.utils.aliases import (
    DocumentsFrame,
    EmbeddingsFrame,
    ScoresFrame,
    TokenIdsFrame,
    TokensFrame,
)
from readnext.utils.io import read_df_from_parquet


# SECTION: Local Data
@pytest.fixture(scope="session")
def documents_frame() -> DocumentsFrame:
    return read_df_from_parquet(DataPaths.merged.documents_frame)


@pytest.fixture(scope="session")
def spacy_tokens_frame() -> TokensFrame:
    return read_df_from_parquet(ResultsPaths.language_models.spacy_tokens_frame_parquet)


@pytest.fixture(scope="session")
def bert_token_ids_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_token_ids_frame_parquet)


@pytest.fixture(scope="session")
def scibert_token_ids_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_token_ids_frame_parquet)


@pytest.fixture(scope="session")
def longformer_token_ids_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.longformer_token_ids_frame_parquet)


@pytest.fixture(scope="session")
def tfidf_embeddings_frame() -> EmbeddingsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.tfidf_embeddings_frame_parquet)


@pytest.fixture(scope="session")
def bm25_embeddings_frame() -> EmbeddingsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bm25_embeddings_frame_parquet)


@pytest.fixture(scope="session")
def word2vec_embeddings_frame() -> EmbeddingsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.word2vec_embeddings_frame_parquet)


@pytest.fixture(scope="session")
def glove_embeddings_frame() -> EmbeddingsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.glove_embeddings_frame_parquet)


@pytest.fixture(scope="session")
def fasttext_embeddings_frame() -> EmbeddingsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.fasttext_embeddings_frame_parquet)


@pytest.fixture(scope="session")
def bert_embeddings_frame() -> EmbeddingsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_embeddings_frame_parquet)


@pytest.fixture(scope="session")
def scibert_embeddings_frame() -> EmbeddingsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_embeddings_frame_parquet)


@pytest.fixture(scope="session")
def longformer_embeddings_frame() -> EmbeddingsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.longformer_embeddings_frame_parquet)


@pytest.fixture(scope="session")
def co_citation_analysis_scores() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.citation_models.co_citation_analysis_scores_parquet)


@pytest.fixture(scope="session")
def bibliographic_coupling_scores() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.citation_models.bibliographic_coupling_scores_parquet)


@pytest.fixture(scope="session")
def tfidf_cosine_similarities() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.language_models.tfidf_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def bm25_cosine_similarities() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bm25_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def word2vec_cosine_similarities() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.language_models.word2vec_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def glove_cosine_similarities() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.language_models.glove_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def fasttext_cosine_similarities() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.language_models.fasttext_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def bert_cosine_similarities() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def scibert_cosine_similarities() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def longformer_cosine_similarities() -> ScoresFrame:
    return read_df_from_parquet(ResultsPaths.language_models.longformer_cosine_similarities_parquet)
