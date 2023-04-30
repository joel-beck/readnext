import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

from readnext.config import ResultsPaths
from readnext.utils import load_df_from_pickle, load_object_from_pickle


@pytest.fixture(scope="module")
def tfidf_embeddings_mapping_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(
        ResultsPaths.language_models.tfidf_embeddings_mapping_most_cited_pkl
    )


@pytest.fixture(scope="module")
def word2vec_embeddings_mapping_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(
        ResultsPaths.language_models.word2vec_embeddings_mapping_most_cited_pkl
    )


@pytest.fixture(scope="module")
def fasttext_embeddings_mapping_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(
        ResultsPaths.language_models.fasttext_embeddings_mapping_most_cited_pkl
    )


@pytest.fixture(scope="module")
def bert_embeddings_mapping_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(ResultsPaths.language_models.bert_embeddings_mapping_most_cited_pkl)


@pytest.fixture(scope="module")
def scibert_embeddings_mapping_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.scibert_embeddings_mapping_most_cited_pkl
    )


def test_tfidf_embeddings_most_cited(
    tfidf_embeddings_mapping_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(tfidf_embeddings_mapping_most_cited, pd.DataFrame)

    # check number and names of columns
    assert tfidf_embeddings_mapping_most_cited.shape[1] == 2
    assert tfidf_embeddings_mapping_most_cited.columns.tolist() == ["document_id", "embedding"]

    # check dtypes of columns
    assert is_integer_dtype(tfidf_embeddings_mapping_most_cited["document_id"])

    first_document = tfidf_embeddings_mapping_most_cited.iloc[0]
    assert isinstance(first_document["embedding"], np.ndarray)
    # only embedding of type float64 instead of float32
    assert first_document["embedding"].dtype == np.float64

    # check embedding dimension for all documents
    assert all(
        len(embedding) == 9166 for embedding in tfidf_embeddings_mapping_most_cited["embedding"]
    )


def test_word2vec_embeddings_most_cited(
    word2vec_embeddings_mapping_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(word2vec_embeddings_mapping_most_cited, pd.DataFrame)

    # check number and names of columns
    assert word2vec_embeddings_mapping_most_cited.shape[1] == 2
    assert word2vec_embeddings_mapping_most_cited.columns.tolist() == ["document_id", "embedding"]

    # check dtypes of columns
    assert is_integer_dtype(word2vec_embeddings_mapping_most_cited["document_id"])

    first_document = word2vec_embeddings_mapping_most_cited.iloc[0]
    assert isinstance(first_document["embedding"], np.ndarray)
    assert first_document["embedding"].dtype == np.float32

    # check embedding dimension
    assert all(
        len(embedding) == 300 for embedding in word2vec_embeddings_mapping_most_cited["embedding"]
    )


def test_fasttext_embeddings_most_cited(
    fasttext_embeddings_mapping_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(fasttext_embeddings_mapping_most_cited, pd.DataFrame)

    # check number and names of columns
    assert fasttext_embeddings_mapping_most_cited.shape[1] == 2
    assert fasttext_embeddings_mapping_most_cited.columns.tolist() == ["document_id", "embedding"]

    # check dtypes of columns
    assert is_integer_dtype(fasttext_embeddings_mapping_most_cited["document_id"])

    first_document = fasttext_embeddings_mapping_most_cited.iloc[0]
    assert isinstance(first_document["embedding"], np.ndarray)
    assert first_document["embedding"].dtype == np.float32

    # check embedding dimension
    assert all(
        len(embedding) == 300 for embedding in fasttext_embeddings_mapping_most_cited["embedding"]
    )


def test_bert_embeddings_most_cited(
    bert_embeddings_mapping_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(bert_embeddings_mapping_most_cited, pd.DataFrame)

    # check number and names of columns
    assert bert_embeddings_mapping_most_cited.shape[1] == 2
    assert bert_embeddings_mapping_most_cited.columns.tolist() == ["document_id", "embedding"]

    # check dtypes of columns
    assert is_integer_dtype(bert_embeddings_mapping_most_cited["document_id"])

    first_document = bert_embeddings_mapping_most_cited.iloc[0]
    assert isinstance(first_document["embedding"], np.ndarray)
    assert first_document["embedding"].dtype == np.float32

    # check embedding dimension
    assert all(
        len(embedding) == 768 for embedding in bert_embeddings_mapping_most_cited["embedding"]
    )


def test_scibert_embeddings_most_cited(
    scibert_embeddings_mapping_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(scibert_embeddings_mapping_most_cited, pd.DataFrame)

    # check number and names of columns
    assert scibert_embeddings_mapping_most_cited.shape[1] == 2
    assert scibert_embeddings_mapping_most_cited.columns.tolist() == ["document_id", "embedding"]

    # check dtypes of columns
    assert is_integer_dtype(scibert_embeddings_mapping_most_cited["document_id"])

    first_document = scibert_embeddings_mapping_most_cited.iloc[0]
    assert isinstance(first_document["embedding"], np.ndarray)
    assert first_document["embedding"].dtype == np.float32

    # check embedding dimension
    assert all(
        len(embedding) == 768 for embedding in scibert_embeddings_mapping_most_cited["embedding"]
    )