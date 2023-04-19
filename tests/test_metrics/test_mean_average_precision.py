import numpy as np
import pandas as pd
import pytest

from readnext.evaluation.metrics import mean_average_precision


def test_mean_average_precision_empty_lists() -> None:
    assert mean_average_precision([]) == 0.0


def test_mean_average_precision_single_empty_list() -> None:
    assert mean_average_precision([[]]) == 0.0


def test_mean_average_precision_single_list_all_zeros() -> None:
    assert mean_average_precision([[0, 0, 0]]) == 0.0


def test_mean_average_precision_single_list_all_ones() -> None:
    assert mean_average_precision([[1, 1, 1]]) == 1.0


def test_mean_average_precision_single_list_with_zeros_and_ones() -> None:
    assert mean_average_precision([[0, 1, 0, 0, 1]]) == 0.45


def test_mean_average_precision_multiple_lists_all_zeros() -> None:
    assert mean_average_precision([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 0.0


def test_mean_average_precision_multiple_lists_all_ones() -> None:
    assert mean_average_precision([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == 1.0


def test_mean_average_precision_multiple_lists_mixed_zeros_and_ones() -> None:
    assert mean_average_precision([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0]]) == pytest.approx(
        0.6851851851851851
    )


def test_mean_average_precision_multiple_lists_different_lengths() -> None:
    assert mean_average_precision([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0, 0]]) == pytest.approx(
        0.6851851851851851
    )


def test_mean_average_precision_with_pandas_series() -> None:
    s1 = pd.Series([0, 1, 0, 0, 1])
    s2 = pd.Series([1, 1, 0, 1, 0])
    assert mean_average_precision([s1, s2]) == pytest.approx(0.683333333333333)


def test_mean_average_precision_with_numpy_arrays() -> None:
    a1 = np.array([0, 1, 0, 0, 1])
    a2 = np.array([1, 1, 0, 1, 0])
    assert mean_average_precision([a1, a2]) == pytest.approx(0.683333333333333)
