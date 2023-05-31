import polars as pl
import pytest

from readnext.config import DataPaths, ResultsPaths
from readnext.modeling.language_models import TokensIdMapping, TokensMapping
from readnext.utils import read_df_from_parquet, read_object_from_pickle


# SECTION: Local Data
@pytest.fixture(scope="session")
def documents_authors_labels_citations_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        DataPaths.merged.documents_authors_labels_citations_most_cited_parquet
    )


@pytest.fixture(scope="session")
def spacy_tokenized_abstracts_mapping_most_cited() -> TokensMapping:
    return read_object_from_pickle(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="session")
def bert_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return read_object_from_pickle(
        ResultsPaths.language_models.bert_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="session")
def scibert_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return read_object_from_pickle(
        ResultsPaths.language_models.scibert_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="session")
def longformer_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return read_object_from_pickle(
        ResultsPaths.language_models.longformer_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="session")
def tfidf_embeddings_most_cited() -> pl.DataFrame:
    return read_object_from_pickle(ResultsPaths.language_models.tfidf_embeddings_most_cited_parquet)


@pytest.fixture(scope="session")
def bm25_embeddings_most_cited() -> pl.DataFrame:
    return read_object_from_pickle(ResultsPaths.language_models.bm25_embeddings_most_cited_parquet)


@pytest.fixture(scope="session")
def word2vec_embeddings_most_cited() -> pl.DataFrame:
    return read_object_from_pickle(
        ResultsPaths.language_models.word2vec_embeddings_most_cited_parquet
    )


@pytest.fixture(scope="session")
def glove_embeddings_most_cited() -> pl.DataFrame:
    return read_object_from_pickle(ResultsPaths.language_models.glove_embeddings_most_cited_parquet)


@pytest.fixture(scope="session")
def fasttext_embeddings_most_cited() -> pl.DataFrame:
    return read_object_from_pickle(
        ResultsPaths.language_models.fasttext_embeddings_most_cited_parquet
    )


@pytest.fixture(scope="session")
def bert_embeddings_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_embeddings_most_cited_parquet)


@pytest.fixture(scope="session")
def scibert_embeddings_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_embeddings_most_cited_parquet)


@pytest.fixture(scope="session")
def longformer_embeddings_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.longformer_embeddings_most_cited_parquet
    )


@pytest.fixture(scope="session")
def co_citation_analysis_scores_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_parquet
    )


@pytest.fixture(scope="session")
def bibliographic_coupling_scores_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_parquet
    )


@pytest.fixture(scope="session")
def tfidf_cosine_similarities_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_parquet
    )


@pytest.fixture(scope="session")
def bm25_cosine_similarities_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.bm25_cosine_similarities_most_cited_parquet
    )


@pytest.fixture(scope="session")
def word2vec_cosine_similarities_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_parquet
    )


@pytest.fixture(scope="session")
def glove_cosine_similarities_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.glove_cosine_similarities_most_cited_parquet
    )


@pytest.fixture(scope="session")
def fasttext_cosine_similarities_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_parquet
    )


@pytest.fixture(scope="session")
def bert_cosine_similarities_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_parquet
    )


@pytest.fixture(scope="session")
def scibert_cosine_similarities_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_parquet
    )


@pytest.fixture(scope="session")
def longformer_cosine_similarities_most_cited() -> pl.DataFrame:
    return read_df_from_parquet(
        ResultsPaths.language_models.longformer_cosine_similarities_most_cited_parquet
    )
