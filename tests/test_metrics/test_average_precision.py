import numpy as np
import pandas as pd
import pytest

from readnext.evaluation.metrics import average_precision


def test_empty_list() -> None:
    assert average_precision([]) == 0.0


def test_all_zeros() -> None:
    assert average_precision([0, 0, 0]) == 0.0


def test_all_ones() -> None:
    assert average_precision([1, 1, 1]) == 1.0


def test_mixed_01() -> None:
    assert average_precision([0, 1, 0, 1, 1]) == pytest.approx(0.533333)


def test_mixed_10() -> None:
    assert average_precision([1, 0, 1, 0, 0]) == pytest.approx(0.8333333)


def test_repeated_labels() -> None:
    assert average_precision([1, 1, 1, 0, 0]) == 1.0


def test_numpy_array_instead_of_list() -> None:
    assert average_precision(np.array([0, 1, 0, 1, 1])) == pytest.approx(0.533333)


def test_pandas_series_instead_of_list() -> None:
    assert average_precision(pd.Series([0, 1, 0, 1, 1])) == pytest.approx(0.5333333)


def test_average_precision_empty_list() -> None:
    assert average_precision([]) == 0.0


def test_average_precision_single_item() -> None:
    assert average_precision([1]) == 1.0
    assert average_precision([0]) == 0.0


def test_average_precision_numpy_array() -> None:
    assert average_precision(np.array([1, 0, 1])) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_pandas_series() -> None:
    assert average_precision(pd.Series([1, 0, 1, 1])) == pytest.approx(0.8055555555)


def test_average_precision_all_zeros() -> None:
    assert average_precision([0, 0, 0, 0, 0]) == 0.0


def test_average_precision_all_ones() -> None:
    assert average_precision([1, 1, 1, 1, 1]) == 1.0


def test_average_precision_alternating_zeros_ones() -> None:
    assert average_precision([0, 1, 0, 1, 0]) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert average_precision([1, 0, 1, 0, 1]) == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)


def test_average_precision_half_zeros_half_ones() -> None:
    assert average_precision([0, 0, 1, 1]) == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_average_precision_numpy_array_all_zeros() -> None:
    assert average_precision(np.array([0, 0, 0, 0, 0])) == 0.0


def test_average_precision_numpy_array_all_ones() -> None:
    assert average_precision(np.array([1, 1, 1, 1, 1])) == 1.0


def test_average_precision_pandas_series_all_zeros() -> None:
    assert average_precision(pd.Series([0, 0, 0, 0, 0])) == 0.0


def test_average_precision_pandas_series_all_ones() -> None:
    assert average_precision(pd.Series([1, 1, 1, 1, 1])) == 1.0


def test_average_precision_mixed_inputs() -> None:
    assert average_precision([1, 0, 1, 1, 0]) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
    assert average_precision(np.array([1, 0, 1, 1, 0])) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
    assert average_precision(pd.Series([1, 0, 1, 1, 0])) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
