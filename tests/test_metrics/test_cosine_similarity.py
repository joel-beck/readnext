import numpy as np
import pytest

from readnext.evaluation.metrics import MismatchingDimensionsError, cosine_similarity


def test_cosine_similarity_basic() -> None:
    u = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    assert cosine_similarity(u, v) == pytest.approx(0.9746318)


def test_cosine_similarity_same_vectors() -> None:
    u = np.array([1, 2, 3])
    assert cosine_similarity(u, u) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    assert cosine_similarity(u, v) == pytest.approx(0.0)


def test_cosine_similarity_perfect_negative_correlation() -> None:
    u = np.array([1, 2, 3])
    v = np.array([-1, -2, -3])
    assert cosine_similarity(u, v) == pytest.approx(-1.0)


def test_cosine_similarity_negative_values() -> None:
    u = np.array([1, -2, 3])
    v = np.array([-4, 5, -6])
    assert cosine_similarity(u, v) == pytest.approx(-0.97463184)


def test_cosine_similarity_different_lengths() -> None:
    u = np.array([1, 2, 3])
    v = np.array([4, 5])
    with pytest.raises(MismatchingDimensionsError):
        cosine_similarity(u, v)


def test_cosine_similarity_input_not_sequence() -> None:
    u = 1
    v = [4, 5, 6]
    with pytest.raises(TypeError):
        cosine_similarity(u, v)  # type: ignore


def test_cosine_similarity_input_not_numeric() -> None:
    u = np.array([1, 2, 3])
    v = np.array(["4", "5", "6"])
    with pytest.raises(ValueError):  # noqa: PT011
        cosine_similarity(u, v)
