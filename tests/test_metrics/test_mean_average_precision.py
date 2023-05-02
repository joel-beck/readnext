import numpy as np
import pandas as pd
import pytest

from readnext.evaluation.metrics import AveragePrecision


def test_mean_average_precision_empty_lists() -> None:
    assert AveragePrecision.mean_average_precision([]) == 0.0


def test_mean_average_precision_single_empty_list() -> None:
    assert AveragePrecision.mean_average_precision([[]]) == 0.0


def test_mean_average_precision_single_list_all_zeros() -> None:
    assert AveragePrecision.mean_average_precision([[0, 0, 0]]) == 0.0


def test_mean_average_precision_single_list_all_ones() -> None:
    assert AveragePrecision.mean_average_precision([[1, 1, 1]]) == 1.0


def test_mean_average_precision_single_list_with_zeros_and_ones() -> None:
    assert AveragePrecision.mean_average_precision([[0, 1, 0, 0, 1]]) == 0.45


def test_mean_average_precision_multiple_lists_all_zeros() -> None:
    assert AveragePrecision.mean_average_precision([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 0.0


def test_mean_average_precision_multiple_lists_all_ones() -> None:
    assert AveragePrecision.mean_average_precision([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == 1.0


def test_mean_average_precision_multiple_lists_mixed_zeros_and_ones() -> None:
    assert AveragePrecision.mean_average_precision(
        [[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0]]
    ) == pytest.approx(0.6851851851851851)


def test_mean_average_precision_multiple_lists_different_lengths() -> None:
    assert AveragePrecision.mean_average_precision(
        [[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0, 0]]
    ) == pytest.approx(0.6851851851851851)


def test_mean_average_precision_with_pandas_series() -> None:
    s1 = pd.Series([0, 1, 0, 0, 1])
    s2 = pd.Series([1, 1, 0, 1, 0])
    assert AveragePrecision.mean_average_precision([s1, s2]) == pytest.approx(0.683333333333333)


def test_mean_average_precision_with_numpy_arrays() -> None:
    a1 = np.array([0, 1, 0, 0, 1])
    a2 = np.array([1, 1, 0, 1, 0])
    assert AveragePrecision.mean_average_precision([a1, a2]) == pytest.approx(0.683333333333333)


def test_mean_average_precision_empty_list() -> None:
    assert AveragePrecision.mean_average_precision([[], []]) == 0.0


def test_mean_average_precision_single_item() -> None:
    assert AveragePrecision.mean_average_precision([[1], [0]]) == 0.5


def test_mean_average_precision_numpy_arrays() -> None:
    input_arrays = [np.array([1, 0, 1]), np.array([1, 0, 1, 1])]
    assert AveragePrecision.mean_average_precision(input_arrays) == pytest.approx(0.819444)


def test_mean_average_precision_pandas_series() -> None:
    input_series = [pd.Series([1, 0, 1]), pd.Series([1, 0, 1, 1])]
    assert AveragePrecision.mean_average_precision(input_series) == pytest.approx(0.819444)


def test_mean_average_precision_all_zeros() -> None:
    input_lists = [[0, 0, 0, 0], [0, 0, 0, 0, 0]]
    assert AveragePrecision.mean_average_precision(input_lists) == 0.0


def test_mean_average_precision_all_ones() -> None:
    input_lists = [[1, 1, 1, 1], [1, 1, 1, 1, 1]]
    assert AveragePrecision.mean_average_precision(input_lists) == 1.0


def test_mean_average_precision_alternating_zeros_ones() -> None:
    input_lists = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
    assert AveragePrecision.mean_average_precision(input_lists) == pytest.approx(
        ((1 / 2 + 2 / 4) / 2 + (1 + 2 / 3 + 3 / 5) / 3) / 2
    )


def test_mean_average_precision_half_zeros_half_ones() -> None:
    input_lists = [[0, 0, 1, 1], [1, 1, 0, 0]]
    assert AveragePrecision.mean_average_precision(input_lists) == pytest.approx(
        ((1 / 3 + 2 / 4) / 2 + (1 + 1) / 2) / 2
    )


def test_mean_average_precision_numpy_array_all_zeros() -> None:
    input_arrays = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0, 0])]
    assert AveragePrecision.mean_average_precision(input_arrays) == 0.0


def test_mean_average_precision_numpy_array_all_ones() -> None:
    input_arrays = [np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1, 1])]
    assert AveragePrecision.mean_average_precision(input_arrays) == 1.0


def test_mean_average_precision_pandas_series_all_zeros() -> None:
    input_series = [pd.Series([0, 0, 0, 0]), pd.Series([0, 0, 0, 0, 0])]
    assert AveragePrecision.mean_average_precision(input_series) == 0.0


def test_mean_average_precision_pandas_series_all_ones() -> None:
    input_series = [pd.Series([1, 1, 1, 1]), pd.Series([1, 1, 1, 1, 1])]
    assert AveragePrecision.mean_average_precision(input_series) == 1.0


def test_mean_average_precision_mixed_inputs() -> None:
    input_lists = [[1, 0, 1, 1, 0], [1, 0, 1, 0, 1]]
    assert AveragePrecision.mean_average_precision(input_lists) == pytest.approx(
        ((1 + 2 / 3 + 3 / 4) / 3 + (1 + 2 / 3 + 3 / 5) / 3) / 2
    )
    input_arrays = [np.array([1, 0, 1, 1, 0]), np.array([1, 0, 1, 0, 1])]
    assert AveragePrecision.mean_average_precision(input_arrays) == pytest.approx(
        ((1 + 2 / 3 + 3 / 4) / 3 + (1 + 2 / 3 + 3 / 5) / 3) / 2
    )
    input_series = [pd.Series([1, 0, 1, 1, 0]), pd.Series([1, 0, 1, 0, 1])]
    assert AveragePrecision.mean_average_precision(input_series) == pytest.approx(
        ((1 + 2 / 3 + 3 / 4) / 3 + (1 + 2 / 3 + 3 / 5) / 3) / 2
    )
