from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd

from readnext.evaluation.metrics import average_precision
from readnext.modeling import CitationModelData, LanguageModelData, ModelData

T = TypeVar("T", bound=ModelData)


# class ScoringFeature(Enum):
#     publication_date = "publication_date_rank"
#     citationcount_document = "citationcount_document_rank"
#     citationcount_author = "citationcount_author_rank"
#     co_citation_analysis = "co_citation_analysis_rank"
#     bibliographic_coupling = "bibliographic_coupling_rank"
#     weighted = "weighted_rank"

#     def __str__(self) -> str:
#         return self.value

#     @property
#     def is_weighted(self) -> bool:
#         return self == self.weighted  # type: ignore


@dataclass
class FeatureWeights:
    publication_date: float = 1.0
    citationcount_document: float = 1.0
    citationcount_author: float = 1.0
    co_citation_analysis: float = 1.0
    bibliographic_coupling: float = 1.0

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "publication_date_rank": self.publication_date,
                "citationcount_document_rank": self.citationcount_document,
                "citationcount_author_rank": self.citationcount_author,
                "co_citation_analysis_rank": self.co_citation_analysis,
                "bibliographic_coupling_rank": self.bibliographic_coupling,
            }
        )


@dataclass
class ModelScorer(ABC, Generic[T]):
    """
    Use Generic instead of `ModelData` directly to use `ModelData` subclasses in
    `ModelScorer` subclasses without violating the Liskov Substitution Principle.
    """

    @staticmethod
    @abstractmethod
    def select_top_n_ranks(
        model_data: T, feature_weights: FeatureWeights | None = None, n: int = 20
    ) -> pd.DataFrame:
        ...

    @staticmethod
    @abstractmethod
    def score_top_n(
        model_data: T,
        feature_weights: FeatureWeights | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        ...

    @staticmethod
    @abstractmethod
    def display_top_n(
        model_data: T, feature_weights: FeatureWeights | None = None, n: int = 20
    ) -> pd.DataFrame:
        ...

    @staticmethod
    def add_labels(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        return pd.merge(df, labels, left_index=True, right_index=True)

    @staticmethod
    def compute_weighted_rowsums(df: pd.DataFrame, feature_weights: FeatureWeights) -> pd.Series:
        return df.mul(feature_weights.to_series()).sum(axis=1)


@dataclass
class CitationModelScorer(ModelScorer):
    @staticmethod
    def select_top_n_ranks(
        citation_model_data: CitationModelData,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        assert feature_weights is not None, "Specify feature weights to rank by"

        return (
            CitationModelScorer.compute_weighted_rowsums(
                citation_model_data.feature_matrix.dropna(), feature_weights
            )
            .rename("weighted_rank")
            .sort_values()
            .head(n)
            .to_frame()
        )

    @staticmethod
    def add_info_cols(df: pd.DataFrame, info_matrix: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(df, info_matrix, left_index=True, right_index=True)

    @staticmethod
    def display_top_n(
        citation_model_data: CitationModelData,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        return CitationModelScorer.select_top_n_ranks(citation_model_data, feature_weights, n).pipe(
            CitationModelScorer.add_info_cols, citation_model_data.info_matrix
        )

    @staticmethod
    def score_top_n(
        citation_model_data: CitationModelData,
        feature_weights: FeatureWeights | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        top_n_ranks_with_labels = CitationModelScorer.select_top_n_ranks(
            citation_model_data, feature_weights, n
        ).pipe(CitationModelScorer.add_labels, citation_model_data.integer_labels)

        return metric(top_n_ranks_with_labels["label"])  # type: ignore


@dataclass
class LanguageModelScorer(ModelScorer):
    @staticmethod
    def select_top_n_ranks(
        language_model_data: LanguageModelData,
        feature_weights: FeatureWeights | None = None,  # noqa: ARG004
        n: int = 20,
    ) -> pd.DataFrame:
        """The `feature_weights` argument is not used for scoring language models."""
        return language_model_data.cosine_similarity_ranks.sort_values(
            "cosine_similarity_rank"
        ).head(n)

    @staticmethod
    def move_cosine_similarity_first(df: pd.DataFrame) -> pd.DataFrame:
        return df[["cosine_similarity"] + list(df.columns.drop("cosine_similarity"))]

    @staticmethod
    def add_info_cols(df: pd.DataFrame, info_matrix: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.merge(df, info_matrix, left_index=True, right_index=True)
            .drop("cosine_similarity_rank", axis="columns")
            .pipe(LanguageModelScorer.move_cosine_similarity_first)
        )

    @staticmethod
    def display_top_n(
        language_model_data: LanguageModelData,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        return LanguageModelScorer.select_top_n_ranks(language_model_data, feature_weights, n).pipe(
            LanguageModelScorer.add_info_cols, language_model_data.info_matrix
        )

    @staticmethod
    def score_top_n(
        language_model_data: LanguageModelData,
        feature_weights: FeatureWeights | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        top_n_ranks_with_labels = LanguageModelScorer.select_top_n_ranks(
            language_model_data, feature_weights, n
        ).pipe(LanguageModelScorer.add_labels, language_model_data.integer_labels)

        return metric(top_n_ranks_with_labels["label"])  # type: ignore
