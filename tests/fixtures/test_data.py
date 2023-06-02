from pathlib import Path

import polars as pl
import pytest

from readnext.utils import (
    EmbeddingsFrame,
    ScoresFrame,
    TokenIdsFrame,
    TokensFrame,
    read_df_from_parquet,
)


@pytest.fixture(scope="session")
def test_documents_authors_labels_citations(root_path: Path) -> pl.DataFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_documents_data.parquet")


@pytest.fixture(scope="session")
def test_bert_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_bert_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_bert_embeddings(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_bert_embeddings.parquet")


@pytest.fixture(scope="session")
def test_bert_tokenized_abstracts_mapping(root_path: Path) -> TokenIdsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_bert_tokenized_abstracts_mapping.parquet"
    )


@pytest.fixture(scope="session")
def test_bibliographic_coupling_scores(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_bibliographic_coupling_scores.parquet"
    )


@pytest.fixture(scope="session")
def test_bm25_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_bm25_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_bm25_embeddings(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_bm25_embeddings.parquet")


@pytest.fixture(scope="session")
def test_co_citation_analysis_scores(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_co_citation_analysis_scores.parquet"
    )


@pytest.fixture(scope="session")
def test_fasttext_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_fasttext_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_fasttext_embeddings(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_fasttext_embeddings.parquet")


@pytest.fixture(scope="session")
def test_glove_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_glove_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_glove_embeddings(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_glove_embeddings.parquet")


@pytest.fixture(scope="session")
def test_longformer_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_longformer_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_longformer_embeddings(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_longformer_embeddings.parquet")


@pytest.fixture(scope="session")
def test_longformer_tokenized_abstracts_mapping(root_path: Path) -> TokenIdsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_longformer_tokenized_abstracts_mapping.parquet"
    )


@pytest.fixture(scope="session")
def test_scibert_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_scibert_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_scibert_embeddings(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_scibert_embeddings.parquet")


@pytest.fixture(scope="session")
def test_scibert_tokenized_abstracts_mapping(root_path: Path) -> TokenIdsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_scibert_tokenized_abstracts_mapping.parquet"
    )


@pytest.fixture(scope="session")
def test_spacy_tokenized_abstracts_mapping(root_path: Path) -> TokensFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_spacy_tokenized_abstracts_mapping.parquet"
    )


@pytest.fixture(scope="session")
def test_tfidf_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_tfidf_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_tfidf_embeddings(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_tfidf_embeddings.parquet")


@pytest.fixture(scope="session")
def test_word2vec_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "data" / "test_word2vec_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_word2vec_embeddings(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(root_path / "tests" / "data" / "test_word2vec_embeddings.parquet")
