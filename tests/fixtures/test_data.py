from pathlib import Path

import pytest

from readnext.utils.aliases import (
    DocumentsFrame,
    EmbeddingsFrame,
    ScoresFrame,
    TokenIdsFrame,
    TokensFrame,
)
from readnext.utils.io import read_df_from_parquet


@pytest.fixture(scope="session")
def test_documents_frame(root_path: Path) -> DocumentsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_documents_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_bert_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_bert_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_bert_embeddings_frame(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_bert_embeddings_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_bert_token_ids_frame(root_path: Path) -> TokenIdsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_bert_token_ids_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_bibliographic_coupling_scores(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_bibliographic_coupling_scores.parquet"
    )


@pytest.fixture(scope="session")
def test_bm25_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_bm25_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_bm25_embeddings_frame(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_bm25_embeddings_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_co_citation_analysis_scores(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_co_citation_analysis_scores.parquet"
    )


@pytest.fixture(scope="session")
def test_fasttext_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_fasttext_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_fasttext_embeddings_frame(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_fasttext_embeddings_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_glove_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_glove_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_glove_embeddings_frame(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_glove_embeddings_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_longformer_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_longformer_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_longformer_embeddings_frame(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_longformer_embeddings_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_longformer_token_ids_frame(root_path: Path) -> TokenIdsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_longformer_token_ids_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_scibert_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_scibert_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_scibert_embeddings_frame(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_scibert_embeddings_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_scibert_token_ids_frame(root_path: Path) -> TokenIdsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_scibert_token_ids_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_spacy_tokens_frame(root_path: Path) -> TokensFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_spacy_tokens_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_tfidf_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_tfidf_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_tfidf_embeddings_frame(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_tfidf_embeddings_frame.parquet"
    )


@pytest.fixture(scope="session")
def test_word2vec_cosine_similarities(root_path: Path) -> ScoresFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_word2vec_cosine_similarities.parquet"
    )


@pytest.fixture(scope="session")
def test_word2vec_embeddings_frame(root_path: Path) -> EmbeddingsFrame:
    return read_df_from_parquet(
        root_path / "tests" / "testing_data" / "test_word2vec_embeddings_frame.parquet"
    )
