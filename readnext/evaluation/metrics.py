from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd

EmbeddingVector: TypeAlias = Sequence[float]
RecommendationLabelList: TypeAlias = Sequence[int]
RecommendationLabelLists: TypeAlias = Sequence[RecommendationLabelList]


class MismatchingDimensionsError(Exception):
    pass


def check_equal_dimensions(u: EmbeddingVector, v: EmbeddingVector) -> None:
    if len(u) != len(v):
        raise MismatchingDimensionsError(
            f"Length of first input = {len(u)} != {len(v)} = Length of second input"
        )


def cosine_similarity(u: EmbeddingVector, v: EmbeddingVector) -> float:
    """Computes cosine similarity between two one-dimensional sequences"""
    check_equal_dimensions(u, v)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))  # type: ignore


def precision(label_list: RecommendationLabelList) -> float:
    """
    Precision = # of relevant items / # of items

    If the labels are binary 0/1 encoded, this coincides with mean(labels).
    """
    return np.mean(label_list)  # type: ignore


def average_precision(label_list: RecommendationLabelList) -> float:
    """
    AP = (1/r) * sum_{k=1}^{K} P(k) * rel(k)

    K = # of items
    r = # of relevant items
    P(k) = precision at k
    rel(k) = 1 if item k is relevant, 0 otherwise

    If the labels are binary 0/1 encoded, this simplifies to:
    r = sum(labels)
    P(k) = mean(labels[:k])
    rel(k) = labels[k] -> relevance scores = labels
    """

    if isinstance(label_list, pd.Series):
        label_list = label_list.to_list()

    num_relevant_items = sum(label_list)
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
    """
    return np.mean([average_precision(label_list) for label_list in label_lists])  # type: ignore
