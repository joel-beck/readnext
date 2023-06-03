import polars as pl
from polars.testing import assert_series_equal

from readnext.utils import add_rank


def test_add_rank_normal_case() -> None:
    input_series = pl.Series([5, 3, 7, 1, 9])
    expected_output = pl.Series([3, 4, 2, 5, 1], dtype=pl.Float64)

    output_series = add_rank(input_series)
    assert_series_equal(output_series, expected_output)


def test_add_rank_empty_series() -> None:
    input_series = pl.Series(dtype=pl.Int64)
    expected_output = pl.Series(dtype=pl.Float64)

    output_series = add_rank(input_series)
    assert_series_equal(output_series, expected_output)


def test_add_rank_with_duplicate_values() -> None:
    input_series = pl.Series([5, 3, 7, 3, 9])
    expected_output = pl.Series([3, 4.5, 2, 4.5, 1], dtype=pl.Float64)

    output_series = add_rank(input_series)
    assert_series_equal(output_series, expected_output)
