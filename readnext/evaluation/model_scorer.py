from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import pandas as pd

from readnext.evaluation.metrics import average_precision
from readnext.modeling import CitationModelData, LanguageModelData, ModelData

T = TypeVar("T", bound=ModelData)


class ScoringFeature(Enum):
    publication_date = "publication_date_rank"
    citationcount_document = "citationcount_document_rank"
    citationcount_author = "citationcount_author_rank"
    co_citation_analysis = "co_citation_analysis_rank"
    bibliographic_coupling = "bibliographic_coupling_rank"
    weighted = "weighted_rank"

    def __str__(self) -> str:
        return self.value

    @property
    def is_weighted(self) -> bool:
        return self == self.weighted  # type: ignore


@dataclass
class ModelScorer(ABC, Generic[T]):
    """
    Use Generic instead of `ModelData` directly to use `ModelData` subclasses in
    `ModelScorer` subclasses without violating the Liskov Substitution Principle.
    """

    @staticmethod
    @abstractmethod
    def select_top_n_ranks(
        model_data: T, scoring_feature: ScoringFeature | None = None, n: int = 20
    ) -> pd.DataFrame:
        ...

    @staticmethod
    @abstractmethod
    def score_top_n(
        model_data: T,
        scoring_feature: ScoringFeature | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        ...

    @staticmethod
    @abstractmethod
    def display_top_n(
        model_data: T,
        scoring_feature: ScoringFeature | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        ...

    @staticmethod
    def add_labels(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        return pd.merge(df, labels, left_index=True, right_index=True)


@dataclass
class CitationModelScorer(ModelScorer):
    @staticmethod
    def add_feature_rank_cols(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            publication_date_rank=df["publication_date"].rank(ascending=False),
            citationcount_document_rank=df["citationcount_document"].rank(ascending=False),
            citationcount_author_rank=df["citationcount_author"].rank(ascending=False),
        )

    @staticmethod
    def set_missing_publication_dates_to_max_rank(df: pd.DataFrame) -> pd.DataFrame:
        # set publication_date_rank to maxiumum rank (number of documents in dataframe) for
        # documents with missing publication date
        return df.assign(publication_date_rank=df["publication_date_rank"].fillna(len(df)))

    @staticmethod
    def select_top_n_ranks(
        citation_model_data: CitationModelData,
        scoring_feature: ScoringFeature | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        assert scoring_feature is not None, "Specify a scoring feature to rank by"

        # `weighted` option for now computes row sums, i.e. each feature is weighted equally
        if scoring_feature.is_weighted:
            ranks_unsorted = (
                citation_model_data.feature_matrix.dropna().sum(axis=1).rename("weighted_rank")
            )
        else:
            ranks_unsorted = citation_model_data.feature_matrix[scoring_feature.value]

        return ranks_unsorted.sort_values().head(n).to_frame()

    @staticmethod
    def add_info_cols(df: pd.DataFrame, info_matrix: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(df, info_matrix, left_index=True, right_index=True)

    @staticmethod
    def display_top_n(
        citation_model_data: CitationModelData,
        scoring_feature: ScoringFeature | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        return CitationModelScorer.select_top_n_ranks(citation_model_data, scoring_feature, n).pipe(
            CitationModelScorer.add_info_cols, citation_model_data.info_matrix
        )

    @staticmethod
    def score_top_n(
        citation_model_data: CitationModelData,
        scoring_feature: ScoringFeature | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        top_n_ranks_with_labels = CitationModelScorer.select_top_n_ranks(
            citation_model_data, scoring_feature, n
        ).pipe(CitationModelScorer.add_labels, citation_model_data.integer_labels)

        return metric(top_n_ranks_with_labels["label"])  # type: ignore


@dataclass
class LanguageModelScorer(ModelScorer):
    @staticmethod
    def select_top_n_ranks(
        language_model_data: LanguageModelData,
        scoring_feature: ScoringFeature | None = None,  # noqa: ARG002
        n: int = 20,
    ) -> pd.DataFrame:
        """The `scoring_feature` argument is not used for scoring language models."""
        return language_model_data.embedding_ranks.sort_values("cosine_similarity_rank").head(n)

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
        scoring_feature: ScoringFeature | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        return LanguageModelScorer.select_top_n_ranks(language_model_data, scoring_feature, n).pipe(
            LanguageModelScorer.add_info_cols, language_model_data.info_matrix
        )

    @staticmethod
    def score_top_n(
        language_model_data: LanguageModelData,
        scoring_feature: ScoringFeature | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        top_n_ranks_with_labels = LanguageModelScorer.select_top_n_ranks(
            language_model_data, scoring_feature, n
        ).pipe(LanguageModelScorer.add_labels, language_model_data.integer_labels)

        return metric(top_n_ranks_with_labels["label"])  # type: ignore
