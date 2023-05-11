import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from pandas.testing import assert_frame_equal

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
def glove_embeddings_mapping_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(
        ResultsPaths.language_models.glove_embeddings_mapping_most_cited_pkl
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


@pytest.mark.xfail(reason="term document matrix for tfidf is not implemented yet")
def test_tfidf_embeddings_most_cited_embedding_dimension(
    tfidf_embeddings_mapping_most_cited: pd.DataFrame,
) -> None:
    assert all(
        len(embedding) == 2037 for embedding in tfidf_embeddings_mapping_most_cited["embedding"]
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


def test_glove_embeddings_most_cited(
    glove_embeddings_mapping_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(glove_embeddings_mapping_most_cited, pd.DataFrame)

    # check number and names of columns
    assert glove_embeddings_mapping_most_cited.shape[1] == 2
    assert glove_embeddings_mapping_most_cited.columns.tolist() == ["document_id", "embedding"]

    # check dtypes of columns
    assert is_integer_dtype(glove_embeddings_mapping_most_cited["document_id"])

    first_document = glove_embeddings_mapping_most_cited.iloc[0]
    assert isinstance(first_document["embedding"], np.ndarray)
    assert first_document["embedding"].dtype == np.float32

    # check embedding dimension
    assert all(
        len(embedding) == 300 for embedding in glove_embeddings_mapping_most_cited["embedding"]
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


def test_that_test_data_mimics_real_data(
    tfidf_embeddings_mapping_most_cited: pd.DataFrame,
    word2vec_embeddings_mapping_most_cited: pd.DataFrame,
    fasttext_embeddings_mapping_most_cited: pd.DataFrame,
    bert_embeddings_mapping_most_cited: pd.DataFrame,
    scibert_embeddings_mapping_most_cited: pd.DataFrame,
    test_tfidf_embeddings_mapping_most_cited: pd.DataFrame,
    test_word2vec_embeddings_mapping_most_cited: pd.DataFrame,
    test_fasttext_embeddings_mapping_most_cited: pd.DataFrame,
    test_bert_embeddings_mapping_most_cited: pd.DataFrame,
    test_scibert_embeddings_mapping_most_cited: pd.DataFrame,
) -> None:
    assert_frame_equal(
        tfidf_embeddings_mapping_most_cited.head(100), test_tfidf_embeddings_mapping_most_cited
    )

    assert_frame_equal(
        word2vec_embeddings_mapping_most_cited.head(100),
        test_word2vec_embeddings_mapping_most_cited,
    )

    assert_frame_equal(
        fasttext_embeddings_mapping_most_cited.head(100),
        test_fasttext_embeddings_mapping_most_cited,
    )

    assert_frame_equal(
        bert_embeddings_mapping_most_cited.head(100),
        test_bert_embeddings_mapping_most_cited,
    )

    assert_frame_equal(
        scibert_embeddings_mapping_most_cited.head(100),
        test_scibert_embeddings_mapping_most_cited,
    )
