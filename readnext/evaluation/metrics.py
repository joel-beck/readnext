from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Vector: TypeAlias = Sequence | NDArray | pd.Series
EmbeddingVector: TypeAlias = Sequence[float] | NDArray | pd.Series
RecommendationLabelList: TypeAlias = Sequence[int] | NDArray | pd.Series
RecommendationLabelLists: TypeAlias = Sequence[RecommendationLabelList]

PairwiseMetric: TypeAlias = Callable[[pd.DataFrame, int, int], int | float]


class MismatchingDimensionsError(Exception):
    pass


def check_equal_dimensions(vec_1: Vector, vec_2: Vector) -> None:
    if len(vec_1) != len(vec_2):
        raise MismatchingDimensionsError(
            f"Length of first input = {len(vec_1)} != {len(vec_2)} = Length of second input"
        )


def count_common_values(
    value_list_1: list[str],
    value_list_2: list[str],
) -> int:
    return len(set(value_list_1).intersection(value_list_2))


def count_common_values_from_df(
    df: pd.DataFrame,
    colname: Literal["citations", "references"],
    document_id_1: int,
    document_id_2: int,
) -> int:
    # iloc[0] to get the first and only value of the pandas Series
    row_value_list: list[str] = df.loc[df["document_id"] == document_id_1, colname].iloc[0]
    col_value_list: list[str] = df.loc[df["document_id"] == document_id_2, colname].iloc[0]

    return count_common_values(row_value_list, col_value_list)


def count_common_references_from_df(
    df: pd.DataFrame, document_id_1: int, document_id_2: int
) -> int:
    return count_common_values_from_df(df, "references", document_id_1, document_id_2)


def count_common_citations_from_df(df: pd.DataFrame, document_id_1: int, document_id_2: int) -> int:
    return count_common_values_from_df(df, "citations", document_id_1, document_id_2)


def cosine_similarity(u: EmbeddingVector, v: EmbeddingVector) -> float:
    """Computes cosine similarity between two one-dimensional sequences"""
    check_equal_dimensions(u, v)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))  # type: ignore


def cosine_similarity_from_df(
    df: pd.DataFrame,
    document_id_1: int,
    document_id_2: int,
) -> float:
    # iloc[0] to get the first and only value of the pandas Series
    row_embedding: EmbeddingVector = df.loc[df["document_id"] == document_id_1, "embedding"].iloc[0]  # type: ignore # noqa: E501
    col_embedding: EmbeddingVector = df.loc[df["document_id"] == document_id_2, "embedding"].iloc[0]  # type: ignore # noqa: E501

    return cosine_similarity(row_embedding, col_embedding)


def precision(label_list: RecommendationLabelList) -> float:
    """
    Precision = # of relevant items / # of items

    If the labels are binary 0/1 encoded, this coincides with mean(labels).
    Precision of empty list is set to 0.0.
    """
    # cannot be simplified to `if not label_list` since label_list may be a numpy array
    # or pandas series
    if not len(label_list):
        return 0.0

    return np.mean(label_list)  # type: ignore


def average_precision(label_list: RecommendationLabelList) -> float:
    """
    AP = (1/r) * sum_{k=1}^{K} P(k) * rel(k)

    K = # of items r = # of relevant items P(k) = precision at k rel(k) = 1 if item k is
    relevant, 0 otherwise

    If the labels are binary 0/1 encoded, this simplifies to: r = sum(labels) P(k) =
    mean(labels[:k]) rel(k) = labels[k] -> relevance scores = labels

    Average Precision of empty list and list with only zeros (only irrelevant items) is
    set to 0.0.
    """
    if not len(label_list):
        return 0.0

    if isinstance(label_list, pd.Series):
        label_list = label_list.to_list()

    num_relevant_items = sum(label_list)
    if num_relevant_items == 0:
        return 0.0

    relevance_scores = label_list

    precision_scores = []
    for k, _ in enumerate(label_list, 1):
        partial_labels = label_list[:k]
        partial_precision = precision(partial_labels)
        precision_scores.append(partial_precision)

    return (1 / num_relevant_items) * np.dot(precision_scores, relevance_scores)  # type: ignore


def mean_average_precision(label_lists: RecommendationLabelLists) -> float:
    """
    Computes mean average precision for multiple recommendation lists.

    Mean Average Precision of empty input is set to 0.0.
    """
    if not len(label_lists):
        return 0.0

    return np.mean([average_precision(label_list) for label_list in label_lists])  # type: ignore
