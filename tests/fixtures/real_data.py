import polars as pl
import pytest

from readnext.config import DataPaths, ResultsPaths
from readnext.utils import TokenIdsFrame, TokensFrame, read_df_from_parquet


# SECTION: Local Data
@pytest.fixture(scope="session")
def documents_data() -> pl.DataFrame:
    return read_df_from_parquet(DataPaths.merged.documents_data)


@pytest.fixture(scope="session")
def spacy_tokenized_abstracts() -> TokensFrame:
    return read_df_from_parquet(ResultsPaths.language_models.spacy_tokenized_abstracts_parquet)


@pytest.fixture(scope="session")
def bert_tokenized_abstracts() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_tokenized_abstracts_parquet)


@pytest.fixture(scope="session")
def scibert_tokenized_abstracts() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_tokenized_abstracts_parquet)


@pytest.fixture(scope="session")
def longformer_tokenized_abstracts() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.longformer_tokenized_abstracts_parquet)


@pytest.fixture(scope="session")
def tfidf_embeddings() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.tfidf_embeddings_parquet)


@pytest.fixture(scope="session")
def bm25_embeddings() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bm25_embeddings_parquet)


@pytest.fixture(scope="session")
def word2vec_embeddings() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.word2vec_embeddings_parquet)


@pytest.fixture(scope="session")
def glove_embeddings() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.glove_embeddings_parquet)


@pytest.fixture(scope="session")
def fasttext_embeddings() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.fasttext_embeddings_parquet)


@pytest.fixture(scope="session")
def bert_embeddings() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_embeddings_parquet)


@pytest.fixture(scope="session")
def scibert_embeddings() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_embeddings_parquet)


@pytest.fixture(scope="session")
def longformer_embeddings() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.longformer_embeddings_parquet)


@pytest.fixture(scope="session")
def co_citation_analysis_scores() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.citation_models.co_citation_analysis_scores_parquet)


@pytest.fixture(scope="session")
def bibliographic_coupling_scores() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.citation_models.bibliographic_coupling_scores_parquet)


@pytest.fixture(scope="session")
def tfidf_cosine_similarities() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.tfidf_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def bm25_cosine_similarities() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bm25_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def word2vec_cosine_similarities() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.word2vec_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def glove_cosine_similarities() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.glove_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def fasttext_cosine_similarities() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.fasttext_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def bert_cosine_similarities() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def scibert_cosine_similarities() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_cosine_similarities_parquet)


@pytest.fixture(scope="session")
def longformer_cosine_similarities() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.longformer_cosine_similarities_parquet)
