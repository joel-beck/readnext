from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
import pandas as pd

from readnext.utils import EmbeddingVector, Vector

TReturn = TypeVar("TReturn", int, float)


class MismatchingDimensionsError(Exception):
    """Custom exception class when two vectors do not have the same dimensions/length."""


@dataclass
class PairwiseMetric(ABC, Generic[TReturn]):
    """Base class for all pairwise metrics."""

    @staticmethod
    @abstractmethod
    def score(vec_1: Vector, vec_2: Vector) -> TReturn:
        ...

    @staticmethod
    @abstractmethod
    def from_df(df: pd.DataFrame, document_id_1: int, document_id_2: int) -> TReturn:
        ...

    @staticmethod
    def check_equal_dimensions(vec_1: Vector, vec_2: Vector) -> None:
        """Raise exception when two vectors do not have the same dimensions/length."""

        if len(vec_1) != len(vec_2):
            raise MismatchingDimensionsError(
                f"Length of first input = {len(vec_1)} != {len(vec_2)} = Length of second input"
            )

    @staticmethod
    def count_common_values(vec_1: Vector, vec_2: Vector) -> int:
        """Count the number of common values between two vectors."""
        return len(set(vec_1).intersection(vec_2))

    @staticmethod
    def count_common_values_from_df(
        df: pd.DataFrame,
        colname: Literal["citations", "references"],
        document_id_1: int,
        document_id_2: int,
    ) -> int:
        """
        Count the number of common values between two vectors that are extracted from a
        DataFrame.
        """
        # iloc[0] to get the first and only value of the pandas Series
        row_value_list: list[str] = df.loc[df["document_id"] == document_id_1, colname].iloc[0]
        col_value_list: list[str] = df.loc[df["document_id"] == document_id_2, colname].iloc[0]

        return PairwiseMetric.count_common_values(row_value_list, col_value_list)


@dataclass
class CountCommonCitations(PairwiseMetric):
    @staticmethod
    def score(vec_1: Vector, vec_2: Vector) -> int:
        return PairwiseMetric.count_common_values(vec_1, vec_2)

    @staticmethod
    def from_df(df: pd.DataFrame, document_id_1: int, document_id_2: int) -> int:
        """
        Count the number of common citations between two documents that are extracted
        from a DataFrame.
        """
        return PairwiseMetric.count_common_values_from_df(
            df, "citations", document_id_1, document_id_2
        )


@dataclass
class CountCommonReferences(PairwiseMetric):
    @staticmethod
    def score(vec_1: Vector, vec_2: Vector) -> int:
        return PairwiseMetric.count_common_values(vec_1, vec_2)

    @staticmethod
    def from_df(df: pd.DataFrame, document_id_1: int, document_id_2: int) -> int:
        """
        Count the number of common references between two documents that are extracted
        from a DataFrame.
        """
        return PairwiseMetric.count_common_values_from_df(
            df, "references", document_id_1, document_id_2
        )


@dataclass
class CosineSimilarity(PairwiseMetric):
    @staticmethod
    def score(vec_1: Vector, vec_2: Vector) -> float:
        """Compute the cosine similarity between two one-dimensional sequences"""
        PairwiseMetric.check_equal_dimensions(vec_1, vec_2)
        return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))  # type: ignore # noqa: E501

    @staticmethod
    def from_df(df: pd.DataFrame, document_id_1: int, document_id_2: int) -> float:
        """
        Compute the cosine similarity between two document embeddings that are extracted
        from a DataFrame.
        """
        # iloc[0] to get the first and only value of the pandas Series
        row_embedding: EmbeddingVector = df.loc[
            df["document_id"] == document_id_1, "embedding"
        ].iloc[0]
        col_embedding: EmbeddingVector = df.loc[
            df["document_id"] == document_id_2, "embedding"
        ].iloc[0]

        return CosineSimilarity.score(row_embedding, col_embedding)
