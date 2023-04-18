import itertools
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from tqdm import tqdm


def count_common_values_pairwise(
    df: pd.DataFrame,
    colname: Literal["citations", "references"],
    document_id_1: int,
    document_id_2: int,
) -> int:
    # iloc[0] to get the first and only value of the pandas Series
    row_value_list: list[str] = df.loc[df["document_id"] == document_id_1, colname].iloc[0]
    col_value_list: list[str] = df.loc[df["document_id"] == document_id_2, colname].iloc[0]

    return len(set(row_value_list).intersection(col_value_list))


def init_values_df_from_input(
    input_df: pd.DataFrame, first_row: int | None, last_row: int | None
) -> pd.DataFrame:
    """
    Sets index labels and column names to document_ids. Values are initialized with 0.
    Number of rows is passed as input, number of columns is inferred from input_df.
    """
    first_row = 0 if first_row is None else first_row
    last_row = len(input_df) if last_row is None else last_row

    col_document_ids = input_df["document_id"]
    row_document_ids = col_document_ids.iloc[first_row:last_row]

    values = np.zeros((len(row_document_ids), len(col_document_ids)), dtype=np.int_)

    return pd.DataFrame(data=values, index=row_document_ids, columns=col_document_ids)


def fill_upper_triangle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy values from lower triangle to upper triangle (or reversed), excluding diagonal
    values. Assumes that the dataframe has an equal number of rows and columns and is
    initialized with zeros.
    """
    return df + df.T - np.diag(df.to_numpy().diagonal())


def fill_values_df(
    input_df: pd.DataFrame,
    values_df: pd.DataFrame,
    count_pairwise_func: Callable[[pd.DataFrame, int, int], int],
) -> pd.DataFrame:
    document_id_combinations = itertools.product(
        enumerate(values_df.index), enumerate(values_df.columns)
    )
    num_iterations = len(values_df) * len(values_df.columns)

    for (row_index, row_document_id), (col_index, col_document_id) in tqdm(
        document_id_combinations, desc="Progress", total=num_iterations
    ):
        # dataframe is symmetric, we only compute values for the lower triangle and
        # diagonal and then copy them to the upper triangle
        if row_index < col_index:
            continue

        # takes the original dataframe and not the new dataframe as input
        num_common_values = count_pairwise_func(input_df, row_document_id, col_document_id)
        # set value of new dataframe
        values_df.loc[row_document_id, col_document_id] = num_common_values

    return values_df


def compute_values_df(
    input_df: pd.DataFrame,
    count_pairwise_func: Callable[[pd.DataFrame, int, int], int],
    first_row: int | None,
    last_row: int | None,
) -> pd.DataFrame:
    values_df = init_values_df_from_input(input_df, first_row, last_row)
    values_df = fill_values_df(input_df, values_df, count_pairwise_func)

    return fill_upper_triangle(values_df)


def lookup_n_most_common(
    df: pd.DataFrame,
    input_document_id: int,
    colname: Literal["num_common_citations", "num_common_references"],
    n: int | None = None,
) -> pd.Series:
    """
    Input Dataframe is already precomputed with all pairwise counts. Computation takes
    place during training.
    """
    full_ranking = (
        df.loc[input_document_id]
        .sort_values(ascending=False)
        .drop(input_document_id)
        .astype(int)
        .rename(colname)
    )

    return full_ranking if n is None else full_ranking.iloc[:n]


def compute_n_most_common(
    df: pd.DataFrame,
    input_document_id: int,
    colname: Literal["citations", "references"],
    count_pairwise_func: Callable[[pd.DataFrame, int, int], int],
    output_name: Literal["num_common_citations", "num_common_references"],
    n: int | None = None,
) -> pd.Series:
    """
    Input Dataframe is raw data. Only computes pairwise counts for the input document
    with respect to all other documents. Computation takes places during inference.
    """
    full_ranking = (
        df[["document_id", colname]]
        .set_index("document_id")
        .apply(
            lambda x: count_pairwise_func(df, input_document_id, x.name),
            axis=1,
        )
        .sort_values(ascending=False)
        .drop(input_document_id)
        .rename(output_name)
    )

    return full_ranking if n is None else full_ranking.iloc[:n]
