from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np
import polars as pl

from readnext.utils.aliases import EmbeddingVector, Vector

TReturn = TypeVar("TReturn", int, float)


class MismatchingDimensionsError(Exception):
    """Custom exception class when two vectors do not have the same dimensions/length."""


@dataclass
class PairwiseMetric(ABC, Generic[TReturn]):
    """Base class for all pairwise metrics."""

    @classmethod
    @abstractmethod
    def score(cls, vec_1: Vector, vec_2: Vector) -> TReturn:
        ...

    @classmethod
    @abstractmethod
    def from_df(cls, df: pl.DataFrame, document_id_1: int, document_id_2: int) -> TReturn:
        ...

    @staticmethod
    def check_equal_dimensions(vec_1: Vector, vec_2: Vector) -> None:
        """Raise exception when two vectors do not have the same dimensions/length."""

        if len(vec_1) != len(vec_2):
            raise MismatchingDimensionsError(
                f"Length of first input = {len(vec_1)} != {len(vec_2)} = Length of second input"
            )


@dataclass
class CountCommonValues(PairwiseMetric):
    @staticmethod
    def count_common_values(vec_1: Vector, vec_2: Vector) -> int:
        """Count the number of common values between two vectors."""
        return len(set(vec_1).intersection(vec_2))

    @classmethod
    def count_common_values_from_df(
        cls,
        df: pl.DataFrame,
        colname: Literal["citations", "references"],
        document_id_1: int,
        document_id_2: int,
    ) -> int:
        """
        Count the number of common values between two vectors that are extracted from a
        DataFrame.
        """

        row_value_list: list[str] = (
            df.filter(pl.col("d3_document_id") == document_id_1).select(colname).item()
        )
        col_value_list: list[str] = (
            df.filter(pl.col("d3_document_id") == document_id_2).select(colname).item()
        )

        return cls.count_common_values(row_value_list, col_value_list)


@dataclass
class CountCommonCitations(CountCommonValues):
    @classmethod
    def score(cls, vec_1: Vector, vec_2: Vector) -> int:
        return cls.count_common_values(vec_1, vec_2)

    @classmethod
    def from_df(cls, df: pl.DataFrame, document_id_1: int, document_id_2: int) -> int:
        """
        Count the number of common citations between two documents that are extracted
        from a DataFrame.
        """
        return cls.count_common_values_from_df(df, "citations", document_id_1, document_id_2)


@dataclass
class CountCommonReferences(CountCommonValues):
    @classmethod
    def score(cls, vec_1: Vector, vec_2: Vector) -> int:
        return cls.count_common_values(vec_1, vec_2)

    @classmethod
    def from_df(cls, df: pl.DataFrame, document_id_1: int, document_id_2: int) -> int:
        """
        Count the number of common references between two documents that are extracted
        from a DataFrame.
        """
        return cls.count_common_values_from_df(df, "references", document_id_1, document_id_2)


@dataclass
class CosineSimilarity(PairwiseMetric):
    @classmethod
    def score(cls, vec_1: Vector, vec_2: Vector) -> float:
        """Compute the cosine similarity between two one-dimensional sequences"""
        cls.check_equal_dimensions(vec_1, vec_2)
        cosine_similarity = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

        return float(cosine_similarity)

    @classmethod
    def from_df(cls, df: pl.DataFrame, document_id_1: int, document_id_2: int) -> float:
        """
        Compute the cosine similarity between two document embeddings that are extracted
        from a DataFrame.
        """
        row_embedding: EmbeddingVector = (
            df.filter(pl.col("d3_document_id") == document_id_1).select("embedding").item()
        )
        col_embedding: EmbeddingVector = (
            df.filter(pl.col("d3_document_id") == document_id_2).select("embedding").item()
        )

        return cls.score(row_embedding, col_embedding)
