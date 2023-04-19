import numpy as np
import pandas as pd

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
    label_list = []
    assert precision(label_list) == 0.0


def test_precision_with_single_value() -> None:
    label_list = [1]
    assert precision(label_list) == 1.0


def test_precision_with_large_list() -> None:
    label_list = [1] * 1000000
    assert precision(label_list) == 1.0
