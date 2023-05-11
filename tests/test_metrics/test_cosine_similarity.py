import numpy as np
import pandas as pd
import pytest

from readnext.evaluation.metrics import CosineSimilarity, MismatchingDimensionsError


@pytest.fixture
def document_embeddings_df() -> pd.DataFrame:
    data = {
        "document_id": [1, 2, 3, 4],
        "embedding": [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]),
        ],
    }
    return pd.DataFrame(data)


def test_cosine_similarity_identical_vectors() -> None:
    assert CosineSimilarity.score([1, 0, 1, 1], [1, 0, 1, 1]) == pytest.approx(1.0)
    assert CosineSimilarity.score([0, 0, 0, 1], [0, 0, 0, 1]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert CosineSimilarity.score([1, 0, 0, 0], [0, 1, 1, 1]) == 0.0
    assert CosineSimilarity.score([0, 0, 1], [1, 1, 0]) == 0.0


def test_cosine_similarity_opposite_vectors() -> None:
    assert CosineSimilarity.score([1, 2, 3], [-1, -2, -3]) == -1.0
    assert CosineSimilarity.score([2, 0, 0], [-2, 0, 0]) == -1.0


def test_cosine_similarity_arbitrary_vectors() -> None:
    assert CosineSimilarity.score([1, 2, 3], [4, 5, 6]) == pytest.approx(0.9746318)


def test_cosine_similarity_numpy_arrays() -> None:
    assert CosineSimilarity.score(np.array([1, 2, 3]), np.array([4, 5, 6])) == pytest.approx(
        0.9746318
    )


def test_cosine_similarity_pandas_series() -> None:
    assert CosineSimilarity.score(pd.Series([1, 2, 3]), pd.Series([4, 5, 6])) == pytest.approx(
        0.9746318
    )


def test_cosine_similarity_mixed_inputs() -> None:
    assert CosineSimilarity.score([1, 2, 3], np.array([4, 5, 6])) == pytest.approx(0.9746318)


def test_cosine_similarity_mismatching_dimensions_error() -> None:
    with pytest.raises(MismatchingDimensionsError) as exc_info:
        CosineSimilarity.score([1, 2, 3], [4, 5])
    assert str(exc_info.value) == "Length of first input = 3 != 2 = Length of second input"


def test_cosine_similarity_long_input() -> None:
    u = [1, 0, 1, 0] * 2500
    v = [0, 1, 0, 1] * 2500
    assert CosineSimilarity.score(u, v) == 0.0


def test_cosine_similarity_from_df(document_embeddings_df: pd.DataFrame) -> None:
    assert CosineSimilarity.from_df(document_embeddings_df, 1, 2) == pytest.approx(0.9746318)
    assert CosineSimilarity.from_df(document_embeddings_df, 1, 1) == pytest.approx(1.0)
    assert CosineSimilarity.from_df(document_embeddings_df, 3, 4) == pytest.approx(0.0)


def test_cosine_similarity_from_df_non_existent_ids(document_embeddings_df: pd.DataFrame) -> None:
    with pytest.raises(IndexError):
        CosineSimilarity.from_df(document_embeddings_df, 1, 5)

    with pytest.raises(IndexError):
        CosineSimilarity.from_df(document_embeddings_df, 6, 2)


def test_cosine_similarity_from_df_empty_dataframe() -> None:
    empty_df = pd.DataFrame(columns=["document_id", "embedding"])
    with pytest.raises(IndexError):
        CosineSimilarity.from_df(empty_df, 1, 2)
