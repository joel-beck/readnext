from collections.abc import Sequence

import numpy as np
import pandas as pd


def precision(labels: Sequence[int]) -> float:
    """
    Precision = # of relevant items / # of items

    If the labels are binary 0/1 encoded, this coincides with mean(labels).
    """
    return np.mean(labels)  # type: ignore


def average_precision(labels: Sequence[int]) -> float:
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

    if isinstance(labels, pd.Series):
        labels = labels.to_list()

    num_relevant_items = sum(labels)
    relevance_scores = labels

    precision_scores = []
    for k, _ in enumerate(labels, 1):
        partial_labels = labels[:k]
        partial_precision = precision(partial_labels)
        precision_scores.append(partial_precision)

    return (1 / num_relevant_items) * np.dot(precision_scores, relevance_scores)  # type: ignore
