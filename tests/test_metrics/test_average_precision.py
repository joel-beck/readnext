import numpy as np
import pandas as pd
import pytest

from readnext.evaluation.metrics import AveragePrecision


def test_empty_list() -> None:
    assert AveragePrecision.score([]) == 0.0


def test_all_zeros() -> None:
    assert AveragePrecision.score([0, 0, 0]) == 0.0


def test_all_ones() -> None:
    assert AveragePrecision.score([1, 1, 1]) == 1.0


def test_mixed_01() -> None:
    assert AveragePrecision.score([0, 1, 0, 1, 1]) == pytest.approx(0.533333)


def test_mixed_10() -> None:
    assert AveragePrecision.score([1, 0, 1, 0, 0]) == pytest.approx(0.8333333)


def test_repeated_labels() -> None:
    assert AveragePrecision.score([1, 1, 1, 0, 0]) == 1.0


def test_numpy_array_instead_of_list() -> None:
    assert AveragePrecision.score(np.array([0, 1, 0, 1, 1])) == pytest.approx(0.533333)


def test_pandas_series_instead_of_list() -> None:
    assert AveragePrecision.score(pd.Series([0, 1, 0, 1, 1])) == pytest.approx(0.5333333)


def test_average_precision_empty_list() -> None:
    assert AveragePrecision.score([]) == 0.0


def test_average_precision_single_item() -> None:
    assert AveragePrecision.score([1]) == 1.0
    assert AveragePrecision.score([0]) == 0.0


def test_average_precision_numpy_array() -> None:
    assert AveragePrecision.score(np.array([1, 0, 1])) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_pandas_series() -> None:
    assert AveragePrecision.score(pd.Series([1, 0, 1, 1])) == pytest.approx(0.8055555555)


def test_average_precision_all_zeros() -> None:
    assert AveragePrecision.score([0, 0, 0, 0, 0]) == 0.0


def test_average_precision_all_ones() -> None:
    assert AveragePrecision.score([1, 1, 1, 1, 1]) == 1.0


def test_average_precision_alternating_zeros_ones() -> None:
    assert AveragePrecision.score([0, 1, 0, 1, 0]) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert AveragePrecision.score([1, 0, 1, 0, 1]) == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)


def test_average_precision_half_zeros_half_ones() -> None:
    assert AveragePrecision.score([0, 0, 1, 1]) == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_average_precision_numpy_array_all_zeros() -> None:
    assert AveragePrecision.score(np.array([0, 0, 0, 0, 0])) == 0.0


def test_average_precision_numpy_array_all_ones() -> None:
    assert AveragePrecision.score(np.array([1, 1, 1, 1, 1])) == 1.0


def test_average_precision_pandas_series_all_zeros() -> None:
    assert AveragePrecision.score(pd.Series([0, 0, 0, 0, 0])) == 0.0


def test_average_precision_pandas_series_all_ones() -> None:
    assert AveragePrecision.score(pd.Series([1, 1, 1, 1, 1])) == 1.0


def test_average_precision_mixed_inputs() -> None:
    assert AveragePrecision.score([1, 0, 1, 1, 0]) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
    assert AveragePrecision.score(np.array([1, 0, 1, 1, 0])) == pytest.approx(
        (1 + 2 / 3 + 3 / 4) / 3
    )
    assert AveragePrecision.score(pd.Series([1, 0, 1, 1, 0])) == pytest.approx(
        (1 + 2 / 3 + 3 / 4) / 3
    )
