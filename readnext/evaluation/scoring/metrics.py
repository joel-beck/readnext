from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Vector: TypeAlias = Sequence | NDArray | pd.Series
EmbeddingVector: TypeAlias = Sequence[float] | NDArray | pd.Series

IntegerLabelList: TypeAlias = Sequence[int] | NDArray | pd.Series
IntegerLabelLists: TypeAlias = Sequence[IntegerLabelList]

StringLabelList: TypeAlias = Sequence[str] | NDArray | pd.Series
StringLabelLists: TypeAlias = Sequence[StringLabelList]

PairwiseMetric: TypeAlias = Callable[[pd.DataFrame, int, int], int | float]

TLabelList = TypeVar("TLabelList", IntegerLabelList, StringLabelList)
TReturn = TypeVar("TReturn", int, float)


class MismatchingDimensionsError(Exception):
    """Custom exception class when two vectors do not have the same dimensions/length."""


def check_equal_dimensions(vec_1: Vector, vec_2: Vector) -> None:
    """Raise exception when two vectors do not have the same dimensions/length."""

    if len(vec_1) != len(vec_2):
        raise MismatchingDimensionsError(
            f"Length of first input = {len(vec_1)} != {len(vec_2)} = Length of second input"
        )


def count_common_values(
    value_list_1: list[str],
    value_list_2: list[str],
) -> int:
    """Count the number of common values between two lists."""
    return len(set(value_list_1).intersection(value_list_2))


def count_common_values_from_df(
    df: pd.DataFrame,
    colname: Literal["citations", "references"],
    document_id_1: int,
    document_id_2: int,
) -> int:
    """
    Count the number of common values between two lists that are extracted from a
    DataFrame.
    """
    # iloc[0] to get the first and only value of the pandas Series
    row_value_list: list[str] = df.loc[df["document_id"] == document_id_1, colname].iloc[0]
    col_value_list: list[str] = df.loc[df["document_id"] == document_id_2, colname].iloc[0]

    return count_common_values(row_value_list, col_value_list)


def count_common_references_from_df(
    df: pd.DataFrame, document_id_1: int, document_id_2: int
) -> int:
    """Count the number of common references between two documents."""
    return count_common_values_from_df(df, "references", document_id_1, document_id_2)


def count_common_citations_from_df(df: pd.DataFrame, document_id_1: int, document_id_2: int) -> int:
    """Count the number of common citations between two documents."""
    return count_common_values_from_df(df, "citations", document_id_1, document_id_2)


def cosine_similarity(u: EmbeddingVector, v: EmbeddingVector) -> float:
    """Compute the cosine similarity between two one-dimensional sequences"""
    check_equal_dimensions(u, v)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))  # type: ignore


def cosine_similarity_from_df(
    df: pd.DataFrame,
    document_id_1: int,
    document_id_2: int,
) -> float:
    """
    Compute the cosine similarity between two document embddings that are extracted from a
    DataFrame.
    """
    # iloc[0] to get the first and only value of the pandas Series
    row_embedding: EmbeddingVector = df.loc[df["document_id"] == document_id_1, "embedding"].iloc[0]  # type: ignore # noqa: E501
    col_embedding: EmbeddingVector = df.loc[df["document_id"] == document_id_2, "embedding"].iloc[0]  # type: ignore # noqa: E501

    return cosine_similarity(row_embedding, col_embedding)


@dataclass
class Metric(ABC, Generic[TLabelList, TReturn]):
    """Base class for all metrics."""

    @staticmethod
    @abstractmethod
    def score(label_list: TLabelList) -> TReturn:
        ...

    @staticmethod
    @abstractmethod
    def from_df(df: pd.DataFrame) -> TReturn:
        ...


@dataclass
class AveragePrecisionMetric(Metric):
    """Average Precision (AP) metric."""

    @staticmethod
    def precision(label_list: IntegerLabelList) -> float:
        """
        Compute the average precision for a list of integer recommendation labels.

        Precision = # of relevant items / # of items

        If the labels are binary 0/1 encoded, this coincides with mean(labels).
        Precision of empty list is set to 0.0.
        """
        # cannot be simplified to `if not label_list` since label_list may be a numpy array
        # or pandas series
        if not len(label_list):
            return 0.0

        return np.mean(label_list)  # type: ignore

    @staticmethod
    def score(label_list: IntegerLabelList) -> float:
        """
        Compute the average precision for a list of integer recommendation labels.

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
            partial_precision = AveragePrecisionMetric.precision(partial_labels)
            precision_scores.append(partial_precision)

        return (1 / num_relevant_items) * np.dot(precision_scores, relevance_scores)  # type: ignore

    @staticmethod
    def from_df(df: pd.DataFrame) -> float:
        """
        Compute the average precision for a list of integer recommendation labels that are
        contained in a dataframe column.
        """
        return AveragePrecisionMetric.score(df["integer_labels"])


@dataclass
class CountUniqueLabelsMetric(Metric):
    """Count the number of unique labels."""

    @staticmethod
    def score(label_list: StringLabelLists) -> int:
        """
        Count the number of unique labels in a list of labels, where the labels are
        themselves lists of strings.
        """
        return len({label for labels in label_list for label in labels})

    @staticmethod
    def from_df(df: pd.DataFrame) -> int:
        """
        Count the number of unique labels in a list of labels that are contained in a
        dataframe column.
        """
        return CountUniqueLabelsMetric.score(df["arxiv_labels"])  # type: ignore


# def mean_average_precision(label_lists: IntegerLabelLists) -> float:
#     """
#     Computes the mean average precision for multiple integer recommendation label lists.

#     Mean Average Precision of empty input is set to 0.0.
#     """
#     if not len(label_lists):
#         return 0.0

#     return np.mean([average_precision(label_list) for label_list in label_lists])  # type: ignore
