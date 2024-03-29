import numpy as np
import pytest

from readnext.evaluation.metrics import AveragePrecision


def test_precision_empty() -> None:
    label_list: list[int] = []
    assert AveragePrecision.precision(label_list) == pytest.approx(0.0)


def test_precision_single_item() -> None:
    assert AveragePrecision.precision([1]) == 1.0
    assert AveragePrecision.precision([0]) == 0.0


def test_precision_zeros() -> None:
    label_list = [0, 0, 0, 0, 0]
    assert AveragePrecision.precision(label_list) == pytest.approx(0.0)


def test_precision_ones() -> None:
    label_list = [1, 1, 1, 1, 1]
    assert AveragePrecision.precision(label_list) == pytest.approx(1.0)


def test_precision_mixed() -> None:
    label_list = [0, 1, 0, 1, 1]
    assert AveragePrecision.precision(label_list) == pytest.approx(0.6)


def test_precision_numpy_array() -> None:
    label_list = np.array([1, 0, 1, 1, 0])
    assert AveragePrecision.precision(label_list) == pytest.approx(0.6)


def test_precision_tuple() -> None:
    label_list = (0, 1, 0, 1, 1)
    assert AveragePrecision.precision(label_list) == pytest.approx(0.6)
