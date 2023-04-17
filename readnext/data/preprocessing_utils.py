"""Shared functions for preprocessing data and feature engineering."""

import pandas as pd


def add_rank(col: pd.Series) -> pd.Series:
    return col.rank(ascending=False)
