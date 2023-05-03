from pathlib import Path

import pandas as pd
import pytest
import torch

from readnext.utils import load_df_from_pickle, load_object_from_pickle


@pytest.fixture(scope="session")
def root_path() -> Path:
    """Return project root path when pytest is executed from the project root directory."""
    return Path().cwd()


@pytest.fixture(scope="session")
def test_bert_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bert_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bert_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_tokenized_abstracts_mapping_most_cited(root_path: Path) -> dict:
    return torch.load(
        root_path / "tests" / "data" / "test_bert_tokenized_abstracts_mapping_most_cited.pt"
    )


@pytest.fixture(scope="session")
def test_bibliographic_coupling_scores_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bibliographic_coupling_scores_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_co_citation_analysis_scores_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_co_citation_analysis_scores_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_documents_authors_labels_citations_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_documents_authors_labels_citations_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_fasttext_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_fasttext_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_fasttext_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_fasttext_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_scibert_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_scibert_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_tokenized_abstracts_mapping_most_cited(root_path: Path) -> dict:
    return torch.load(
        root_path / "tests" / "data" / "test_scibert_tokenized_abstracts_mapping_most_cited.pt"
    )


@pytest.fixture(scope="session")
def test_spacy_tokenized_abstracts_mapping_most_cited(root_path: Path) -> pd.DataFrame:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_spacy_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_tfidf_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_tfidf_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_tfidf_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_tfidf_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_word2vec_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_word2vec_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_word2vec_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_word2vec_embeddings_mapping_most_cited.pkl"
    )
