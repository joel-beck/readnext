from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

IntegerLabelList: TypeAlias = Sequence[int] | NDArray | pd.Series
IntegerLabelLists: TypeAlias = Sequence[IntegerLabelList]

StringLabelList: TypeAlias = Sequence[str] | NDArray | pd.Series
StringLabelLists: TypeAlias = Sequence[StringLabelList]

TLabelList = TypeVar("TLabelList", IntegerLabelList, StringLabelList)
TReturn = TypeVar("TReturn", int, float)


@dataclass
class EvaluationMetric(ABC, Generic[TLabelList, TReturn]):
    """Base class for all evaluation metrics."""

    @staticmethod
    @abstractmethod
    def score(label_list: TLabelList) -> TReturn:
        ...

    @staticmethod
    @abstractmethod
    def from_df(df: pd.DataFrame) -> TReturn:
        ...


@dataclass
class AveragePrecision(EvaluationMetric):
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
            partial_precision = AveragePrecision.precision(partial_labels)
            precision_scores.append(partial_precision)

        return (1 / num_relevant_items) * np.dot(precision_scores, relevance_scores)  # type: ignore

    @staticmethod
    def mean_average_precision(label_lists: IntegerLabelLists) -> float:
        """
        Computes the mean average precision for multiple integer recommendation label lists.

        Mean Average Precision of empty input is set to 0.0.
        """
        if not len(label_lists):
            return 0.0

        return np.mean([AveragePrecision.score(label_list) for label_list in label_lists])  # type: ignore # noqa: E501

    @staticmethod
    def from_df(df: pd.DataFrame) -> float:
        """
        Compute the average precision for a list of integer recommendation labels that are
        contained in a dataframe column.
        """
        return AveragePrecision.score(df["integer_labels"])


@dataclass
class CountUniqueLabels(EvaluationMetric):
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
        return CountUniqueLabels.score(df["arxiv_labels"])  # type: ignore
