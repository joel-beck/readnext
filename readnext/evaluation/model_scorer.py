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

    @abstractmethod
    def select_top_n_ranks(
        self, model_data: T, by: ScoringFeature | None = None, n: int = 20
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def score_top_n(
        self,
        model_data: T,
        by: ScoringFeature | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        ...

    @abstractmethod
    def display_top_n(
        self,
        model_data: T,
        by: ScoringFeature | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        ...

    def add_labels(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        return pd.merge(df, labels, left_index=True, right_index=True)


@dataclass
class CitationModelScorer(ModelScorer):
    def add_feature_rank_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(
            publication_date_rank=df["publication_date"].rank(ascending=False),
            citationcount_document_rank=df["citationcount_document"].rank(ascending=False),
            citationcount_author_rank=df["citationcount_author"].rank(ascending=False),
        )

    def set_missing_publication_dates_to_max_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        # set publication_date_rank to maxiumum rank (number of documents in dataframe) for
        # documents with missing publication date
        return df.assign(publication_date_rank=df["publication_date_rank"].fillna(len(df)))

    def select_top_n_ranks(
        self, citation_model_data: CitationModelData, by: ScoringFeature | None = None, n: int = 20
    ) -> pd.DataFrame:
        assert by is not None, "Specify a scoring feature to rank by"

        # `weighted` option for now computes row sums, i.e. each feature is weighted equally
        if by.is_weighted:
            ranks_unsorted = (
                citation_model_data.feature_matrix.dropna().sum(axis=1).rename("weighted_rank")
            )
        else:
            ranks_unsorted = citation_model_data.feature_matrix[by.value]

        return ranks_unsorted.sort_values().head(n).to_frame()

    def add_info_cols(self, df: pd.DataFrame, info_matrix: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(df, info_matrix, left_index=True, right_index=True)

    def display_top_n(
        self,
        citation_model_data: CitationModelData,
        by: ScoringFeature | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        return self.select_top_n_ranks(citation_model_data, by, n).pipe(
            self.add_info_cols, citation_model_data.info_matrix
        )

    def score_top_n(
        self,
        citation_model_data: CitationModelData,
        by: ScoringFeature | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        top_n_ranks_with_labels = self.select_top_n_ranks(citation_model_data, by, n).pipe(
            self.add_labels, citation_model_data.integer_labels
        )

        return metric(top_n_ranks_with_labels["label"])  # type: ignore


@dataclass
class LanguageModelScorer(ModelScorer):
    def select_top_n_ranks(
        self,
        language_model_data: LanguageModelData,
        by: ScoringFeature | None = None,  # noqa: ARG002
        n: int = 20,
    ) -> pd.DataFrame:
        """The `by` argument is not used for scoring language models."""
        return language_model_data.embedding_ranks.sort_values("cosine_similarity_rank").head(n)

    def move_cosine_similarity_first(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["cosine_similarity"] + list(df.columns.drop("cosine_similarity"))]

    def add_info_cols(self, df: pd.DataFrame, info_matrix: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.merge(df, info_matrix, left_index=True, right_index=True)
            .drop("cosine_similarity_rank", axis="columns")
            .pipe(self.move_cosine_similarity_first)
        )

    def display_top_n(
        self,
        language_model_data: LanguageModelData,
        by: ScoringFeature | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        return self.select_top_n_ranks(language_model_data, by, n).pipe(
            self.add_info_cols, language_model_data.info_matrix
        )

    def score_top_n(
        self,
        language_model_data: LanguageModelData,
        by: ScoringFeature | None = None,
        metric: Callable[[Sequence[int]], float] = average_precision,
        n: int = 20,
    ) -> float:
        top_n_ranks_with_labels = self.select_top_n_ranks(language_model_data, by, n).pipe(
            self.add_labels, language_model_data.integer_labels
        )

        return metric(top_n_ranks_with_labels["label"])  # type: ignore
