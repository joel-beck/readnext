import numpy as np
import polars as pl
import pytest

from readnext.evaluation.metrics import AveragePrecision


def test_average_precision_score_empty() -> None:
    label_list: list[int] = []
    assert AveragePrecision.score(label_list) == pytest.approx(0.0)


def test_average_precision_score_zeros() -> None:
    label_list = [0, 0, 0, 0, 0]
    assert AveragePrecision.score(label_list) == pytest.approx(0.0)


def test_average_precision_score_ones() -> None:
    label_list = [1, 1, 1, 1, 1]
    assert AveragePrecision.score(label_list) == pytest.approx(1.0)


def test_average_precision_score_mixed() -> None:
    label_list = [0, 1, 0, 1, 1]
    assert AveragePrecision.score(label_list) == pytest.approx(0.533333333)


def test_average_precision_score_numpy_array() -> None:
    label_list = np.array([0, 1, 0, 1, 1])
    assert AveragePrecision.score(label_list) == pytest.approx(0.533333333)


def test_average_precision_score_polars_series() -> None:
    label_list = pl.Series([0, 1, 0, 1, 1])
    assert AveragePrecision.score(label_list) == pytest.approx(0.533333333)


def test_average_precision_from_df() -> None:
    data = {
        "integer_label": [
            *[0, 1, 0, 1, 1],
            *[1, 1, 1, 1, 1],
            *[0, 0, 0, 0, 0],
            *[1, 0, 1, 0, 1],
        ]
    }
    df = pl.DataFrame(data)
    assert AveragePrecision.from_df(df) == pytest.approx(0.6342532467532468)


def test_average_precision_from_df_empty() -> None:
    data: dict[str, list[int]] = {"integer_label": []}
    df = pl.DataFrame(data)
    assert AveragePrecision.from_df(df) == pytest.approx(0.0)


def test_average_precision_from_df_zeros() -> None:
    data = {"integer_label": [0, 0, 0, 0, 0]}
    df = pl.DataFrame(data)
    assert AveragePrecision.from_df(df) == pytest.approx(0.0)


def test_average_precision_from_df_ones() -> None:
    data = {"integer_label": [1, 1, 1, 1, 1]}
    df = pl.DataFrame(data)
    assert AveragePrecision.from_df(df) == pytest.approx(1.0)


def test_average_precision_from_df_mixed() -> None:
    data = {"integer_label": [0, 1, 0, 1, 1]}
    df = pl.DataFrame(data)
    assert AveragePrecision.from_df(df) == pytest.approx(0.533333333)


def test_average_precision_from_df_tuple() -> None:
    data = {"integer_label": (0, 1, 0, 1, 1)}
    df = pl.DataFrame(data)
    assert AveragePrecision.from_df(df) == pytest.approx(0.533333333)


def test_average_precision_from_df_series_of_integers() -> None:
    data = {"integer_label": pl.Series([0, 1, 0, 1, 1])}
    df = pl.DataFrame(data)
    assert AveragePrecision.from_df(df) == pytest.approx(0.533333333)
