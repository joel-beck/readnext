import itertools
from typing import Literal

import numpy as np
import pandas as pd

from readnext.evaluation.metrics import (
    PairwiseMetric,
    cosine_similarity_from_df,
    count_common_citations_from_df,
    count_common_references_from_df,
)
from readnext.utils import setup_progress_bar


def init_pairwise_scores_df_from_input(
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

    output_values = np.zeros((len(row_document_ids), len(col_document_ids)), dtype=np.int_)

    return pd.DataFrame(data=output_values, index=row_document_ids, columns=col_document_ids)


def fill_pairwise_scores_df(
    input_df: pd.DataFrame,
    pairwise_scores_df: pd.DataFrame,
    pairwise_metric: PairwiseMetric,
) -> pd.DataFrame:
    document_id_combinations = itertools.product(
        enumerate(pairwise_scores_df.index), enumerate(pairwise_scores_df.columns)
    )
    num_iterations = len(pairwise_scores_df) * len(pairwise_scores_df.columns)

    with setup_progress_bar() as progress_bar:
        for (row_index, row_document_id), (col_index, col_document_id) in progress_bar.track(
            document_id_combinations, total=num_iterations
        ):
            # dataframe is symmetric, we only compute values for the lower triangle and
            # diagonal and then copy them to the upper triangle
            if row_index < col_index:
                continue

            # takes the original dataframe and not the new dataframe as input
            pairwise_score = pairwise_metric(input_df, row_document_id, col_document_id)
            # set value of new dataframe
            pairwise_scores_df.loc[row_document_id, col_document_id] = pairwise_score

    return pairwise_scores_df


def fill_upper_triangle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy values from lower triangle to upper triangle (or reversed), excluding diagonal
    values. Assumes that the dataframe has an equal number of rows and columns and is
    initialized with zeros.
    """
    return df + df.T - np.diag(df.to_numpy().diagonal())


def precompute_pairwise_scores(
    input_df: pd.DataFrame,
    pairwise_metric: PairwiseMetric,
    first_row: int | None,
    last_row: int | None,
) -> pd.DataFrame:
    pairwise_scores_df = init_pairwise_scores_df_from_input(input_df, first_row, last_row)
    pairwise_scores_df = fill_pairwise_scores_df(input_df, pairwise_scores_df, pairwise_metric)

    return fill_upper_triangle(pairwise_scores_df)


def precompute_co_citations(
    df: pd.DataFrame, first_row: int | None = None, last_row: int | None = None
) -> pd.DataFrame:
    return precompute_pairwise_scores(df, count_common_citations_from_df, first_row, last_row)


def precompute_co_references(
    df: pd.DataFrame, first_row: int | None = None, last_row: int | None = None
) -> pd.DataFrame:
    return precompute_pairwise_scores(df, count_common_references_from_df, first_row, last_row)


def precompute_cosine_similarities(
    df: pd.DataFrame, first_row: int | None = None, last_row: int | None = None
) -> pd.DataFrame:
    return precompute_pairwise_scores(df, cosine_similarity_from_df, first_row, last_row)


def lookup_n_highest_pairwise_scores(
    df: pd.DataFrame,
    input_document_id: int,
    output_colname: Literal["num_common_citations", "num_common_references", "cosine_similarity"],
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
        .rename(output_colname)
    )

    return full_ranking if n is None else full_ranking.iloc[:n]


def lookup_n_most_common_citations(
    df: pd.DataFrame,
    input_document_id: int,
    n: int | None = None,
) -> pd.Series:
    return lookup_n_highest_pairwise_scores(df, input_document_id, "num_common_citations", n)


def lookup_n_most_common_references(
    df: pd.DataFrame,
    input_document_id: int,
    n: int | None = None,
) -> pd.Series:
    return lookup_n_highest_pairwise_scores(df, input_document_id, "num_common_references", n)


def compute_n_highest_pairwise_scores(
    df: pd.DataFrame,
    input_document_id: int,
    colname: Literal["citations", "references"],
    pairwise_metric: PairwiseMetric,
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
            lambda x: pairwise_metric(df, input_document_id, x.name),
            axis=1,
        )
        .sort_values(ascending=False)
        .drop(input_document_id)
        .rename(output_name)
    )

    return full_ranking if n is None else full_ranking.iloc[:n]


def compute_n_most_common_citations(
    df: pd.DataFrame,
    input_document_id: int,
    n: int | None = None,
) -> pd.Series:
    return compute_n_highest_pairwise_scores(
        df,
        input_document_id,
        "citations",
        count_common_references_from_df,
        "num_common_citations",
        n,
    )


def compute_n_most_common_references(
    df: pd.DataFrame,
    input_document_id: int,
    n: int | None = None,
) -> pd.Series:
    return compute_n_highest_pairwise_scores(
        df,
        input_document_id,
        "references",
        count_common_references_from_df,
        "num_common_references",
        n,
    )
