import numpy as np
import pandas as pd
import pytest

from readnext.evaluation.metrics import CosineSimilarity, MismatchingDimensionsError


def test_cosine_similarity_basic() -> None:
    u = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    assert CosineSimilarity.score(u, v) == pytest.approx(0.9746318)


def test_cosine_similarity_same_vectors() -> None:
    u = np.array([1, 2, 3])
    assert CosineSimilarity.score(u, u) == pytest.approx(1.0)


def test_cosine_similarity_perfect_negative_correlation() -> None:
    u = np.array([1, 2, 3])
    v = np.array([-1, -2, -3])
    assert CosineSimilarity.score(u, v) == pytest.approx(-1.0)


def test_cosine_similarity_negative_values() -> None:
    u = np.array([1, -2, 3])
    v = np.array([-4, 5, -6])
    assert CosineSimilarity.score(u, v) == pytest.approx(-0.97463184)


def test_cosine_similarity_different_lengths() -> None:
    u = np.array([1, 2, 3])
    v = np.array([4, 5])
    with pytest.raises(MismatchingDimensionsError):
        CosineSimilarity.score(u, v)


def test_cosine_similarity_input_not_sequence() -> None:
    u = 1
    v = [4, 5, 6]
    with pytest.raises(TypeError):
        CosineSimilarity.score(u, v)  # type: ignore


def test_cosine_similarity_input_not_numeric() -> None:
    u = np.array([1, 2, 3])
    v = np.array(["4", "5", "6"])
    with pytest.raises(ValueError):  # noqa: PT011
        CosineSimilarity.score(u, v)


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
    with pytest.raises(MismatchingDimensionsError):
        CosineSimilarity.score([1, 2, 3], [4, 5])


def test_cosine_similarity_long_input() -> None:
    u = [1, 0, 1, 0] * 2500
    v = [0, 1, 0, 1] * 2500
    assert CosineSimilarity.score(u, v) == 0.0
