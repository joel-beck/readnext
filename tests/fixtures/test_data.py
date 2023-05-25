from pathlib import Path

import pandas as pd
import pytest

from readnext.utils import (
    EmbeddingsMapping,
    ScoresFrame,
    TokensIdMapping,
    TokensMapping,
    load_df_from_pickle,
    load_object_from_pickle,
)


@pytest.fixture(scope="session")
def test_documents_authors_labels_citations_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_documents_authors_labels_citations_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bert_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bert_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_tokenized_abstracts_mapping_most_cited(root_path: Path) -> TokensIdMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bert_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bibliographic_coupling_scores_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bibliographic_coupling_scores_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bm25_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bm25_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bm25_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bm25_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_co_citation_analysis_scores_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_co_citation_analysis_scores_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_fasttext_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_fasttext_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_fasttext_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_fasttext_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_glove_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_glove_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_glove_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_glove_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_longformer_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_longformer_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_longformer_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_longformer_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_longformer_tokenized_abstracts_mapping_most_cited(root_path: Path) -> TokensIdMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_longformer_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_scibert_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_scibert_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_tokenized_abstracts_mapping_most_cited(root_path: Path) -> TokensIdMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_scibert_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_spacy_tokenized_abstracts_mapping_most_cited(root_path: Path) -> TokensMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_spacy_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_tfidf_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_tfidf_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_tfidf_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_tfidf_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_word2vec_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_word2vec_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_word2vec_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_word2vec_embeddings_most_cited.pkl"
    )
