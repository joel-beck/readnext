import numpy as np
import pandas as pd
import pytest

from readnext.evaluation.metrics import precision


def test_precision_with_all_ones() -> None:
    label_list = [1, 1, 1, 1, 1]
    assert precision(label_list) == 1.0


def test_precision_with_all_zeros() -> None:
    label_list = [0, 0, 0, 0, 0]
    assert precision(label_list) == 0.0


def test_precision_with_half_ones() -> None:
    label_list = [1, 1, 0, 0, 0]
    assert precision(label_list) == 0.4


def test_precision_with_half_zeros() -> None:
    label_list = [0, 0, 1, 1, 1]
    assert precision(label_list) == 0.6


def test_precision_with_np_array() -> None:
    label_list = np.array([1, 1, 0, 1, 0])
    assert precision(label_list) == 0.6


def test_precision_with_pd_series() -> None:
    label_list = pd.Series([0, 1, 1, 0, 1])
    assert precision(label_list) == 0.6


def test_precision_with_empty_list() -> None:
    label_list: list[int] = []
    assert precision(label_list) == 0.0


def test_precision_with_single_value() -> None:
    label_list = [1]
    assert precision(label_list) == 1.0


def test_precision_with_large_list() -> None:
    label_list = [1] * 1000000
    assert precision(label_list) == 1.0


def test_precision_empty_list() -> None:
    assert precision([]) == 0.0


def test_precision_single_item() -> None:
    assert precision([1]) == 1.0
    assert precision([0]) == 0.0


def test_precision_numpy_array() -> None:
    assert precision(np.array([1, 0, 1])) == pytest.approx(2 / 3)


def test_precision_pandas_series() -> None:
    assert precision(pd.Series([1, 0, 1, 1])) == pytest.approx(3 / 4)


def test_precision_all_zeros() -> None:
    assert precision([0, 0, 0, 0, 0]) == 0.0


def test_precision_all_ones() -> None:
    assert precision([1, 1, 1, 1, 1]) == 1.0


def test_precision_alternating_zeros_ones() -> None:
    assert precision([0, 1, 0, 1, 0]) == 0.4
    assert precision([1, 0, 1, 0, 1]) == 0.6


def test_precision_half_zeros_half_ones() -> None:
    assert precision([0, 0, 1, 1]) == 0.5


def test_precision_numpy_array_all_zeros() -> None:
    assert precision(np.array([0, 0, 0, 0, 0])) == 0.0


def test_precision_numpy_array_all_ones() -> None:
    assert precision(np.array([1, 1, 1, 1, 1])) == 1.0


def test_precision_pandas_series_all_zeros() -> None:
    assert precision(pd.Series([0, 0, 0, 0, 0])) == 0.0


def test_precision_pandas_series_all_ones() -> None:
    assert precision(pd.Series([1, 1, 1, 1, 1])) == 1.0


def test_precision_mixed_inputs() -> None:
    assert precision([1, 0, 1, 1, 0]) == 0.6
    assert precision(np.array([1, 0, 1, 1, 0])) == pytest.approx(0.6)
    assert precision(pd.Series([1, 0, 1, 1, 0])) == pytest.approx(0.6)


def test_precision_long_input() -> None:
    assert precision([1, 0, 1, 0] * 2500) == 0.5
