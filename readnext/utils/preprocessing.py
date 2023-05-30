"""Shared functions for preprocessing data and feature engineering."""

import polars as pl


def add_rank(col: pl.Series) -> pl.Series:
    """
    Ranks the values in a column in descending order. Ties are resolved by assigning the
    average rank to all tied values.
    """
    return col.rank(descending=True, method="average")
