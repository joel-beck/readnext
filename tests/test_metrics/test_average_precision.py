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
