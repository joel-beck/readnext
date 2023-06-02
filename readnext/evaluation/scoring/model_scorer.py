from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import polars as pl

from readnext.evaluation.metrics import AveragePrecision, CountUniqueLabels
from readnext.modeling import CitationModelData, LanguageModelData, ModelData

TModelData = TypeVar("TModelData", bound=ModelData)


@dataclass
class FeatureWeights:
    """
    Holds the weights for the citation features and global document features for the
    non-language model recommender.
    """

    publication_date: float = 1.0
    citationcount_document: float = 1.0
    citationcount_author: float = 1.0
    co_citation_analysis: float = 1.0
    bibliographic_coupling: float = 1.0


@dataclass
class ModelScorer(ABC, Generic[TModelData]):
    """
    Base class for computing scores and selecting the best recommendations from a model.

    Use a Generic instead of the `ModelData` class directly to use `ModelData`
    subclasses in `ModelScorer` subclasses without violating the Liskov Substitution
    Principle.
    """

    @staticmethod
    @abstractmethod
    def select_top_n_ranks(
        model_data: TModelData, feature_weights: FeatureWeights, n: int
    ) -> pl.DataFrame:
        ...

    @overload
    @staticmethod
    @abstractmethod
    def score_top_n(
        model_data: TModelData,
        metric: AveragePrecision,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> float:
        ...

    @overload
    @staticmethod
    @abstractmethod
    def score_top_n(
        model_data: TModelData,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> int:
        ...

    @staticmethod
    @abstractmethod
    def score_top_n(
        model_data: TModelData,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> float | int:
        ...

    @staticmethod
    @abstractmethod
    def display_top_n(
        model_data: TModelData, feature_weights: FeatureWeights | None = None, n: int = 20
    ) -> pl.DataFrame:
        ...

    @staticmethod
    def add_labels(df: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
        """Add a vector of labels to a dataframe."""
        return df.join(labels, on="candidate_d3_document_id", how="left")

    @staticmethod
    def compute_weighted_rank(
        feature_matrix: pl.DataFrame, feature_weights: FeatureWeights
    ) -> pl.DataFrame:
        """
        Compute the weighted rowsums of a dataframe with one weight for each citation
        feature column. The output dataframe has two columns named `candidate_d3_document_id` and
        `weighted_rank`.
        """

        return feature_matrix.with_columns(
            weighted_rank=(
                feature_weights.publication_date * feature_matrix["publication_date_rank"]
                + feature_weights.citationcount_document
                * feature_matrix["citationcount_document_rank"]
                + feature_weights.citationcount_author * feature_matrix["citationcount_author_rank"]
                + feature_weights.co_citation_analysis * feature_matrix["co_citation_analysis_rank"]
                + feature_weights.bibliographic_coupling
                * feature_matrix["bibliographic_coupling_rank"]
            )
        ).select(["candidate_d3_document_id", "weighted_rank"])


@dataclass
class CitationModelScorer(ModelScorer):
    @staticmethod
    def select_top_n_ranks(
        citation_model_data: CitationModelData, feature_weights: FeatureWeights, n: int
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a citation model in a dataframe.
        """

        return (
            CitationModelScorer.compute_weighted_rank(
                citation_model_data.feature_matrix,
                feature_weights,
            )
            .sort(by="weighted_rank")
            .head(n)
        )

    @staticmethod
    def add_info_cols(df: pl.DataFrame, info_matrix: pl.DataFrame) -> pl.DataFrame:
        """Add document info columns to a dataframe."""
        return df.join(info_matrix, on="candidate_d3_document_id", how="left")

    @staticmethod
    def display_top_n(
        citation_model_data: CitationModelData,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a citation model in a
        dataframe together with additional information columns about the documents.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        return CitationModelScorer.select_top_n_ranks(citation_model_data, feature_weights, n).pipe(
            CitationModelScorer.add_info_cols, citation_model_data.info_matrix
        )

    @overload
    @staticmethod
    def score_top_n(
        citation_model_data: CitationModelData,
        metric: AveragePrecision,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> float:
        ...

    @overload
    @staticmethod
    def score_top_n(
        citation_model_data: CitationModelData,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> int:
        ...

    @staticmethod
    def score_top_n(
        citation_model_data: CitationModelData,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> float | int:
        """
        Compute the average precision (or a different metric) for the top n
        recommendations.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        top_n_ranks_with_labels = CitationModelScorer.display_top_n(
            citation_model_data, feature_weights, n
        ).pipe(CitationModelScorer.add_labels, citation_model_data.integer_labels)

        return metric.from_df(top_n_ranks_with_labels)


@dataclass
class LanguageModelScorer(ModelScorer):
    @staticmethod
    def select_top_n_ranks(
        language_model_data: LanguageModelData,
        feature_weights: FeatureWeights,  # noqa: ARG004
        n: int,
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a language model in a dataframe.

        The `feature_weights` argument is not used for scoring language models.
        """
        return language_model_data.cosine_similarity_ranks.sort("rank").head(n)

    @staticmethod
    def move_cosine_similarity_first(df: pl.DataFrame) -> pl.DataFrame:
        """Move the cosine similarity column to the first position in a dataframe."""
        return df.select(["cosine_similarity", pl.exclude("cosine_similarity")])

    @staticmethod
    def add_info_cols(df: pl.DataFrame, info_matrix: pl.DataFrame) -> pl.DataFrame:
        """
        Add document info columns to a dataframe. The cosine similarity column is moved
        to the first position and replaces the cosine similarity rank column.
        """
        return (
            df.join(info_matrix, on="candidate_d3_document_id", how="left")
            .drop("rank")
            .pipe(LanguageModelScorer.move_cosine_similarity_first)
        )

    @staticmethod
    def display_top_n(
        language_model_data: LanguageModelData,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a language model in a
        dataframe together with additional information columns about the documents.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        return LanguageModelScorer.select_top_n_ranks(language_model_data, feature_weights, n).pipe(
            LanguageModelScorer.add_info_cols, language_model_data.info_matrix
        )

    @overload
    @staticmethod
    def score_top_n(
        language_model_data: LanguageModelData,
        metric: AveragePrecision,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> float:
        ...

    @overload
    @staticmethod
    def score_top_n(
        language_model_data: LanguageModelData,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> int:
        ...

    @staticmethod
    def score_top_n(
        language_model_data: LanguageModelData,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = 20,
    ) -> float | int:
        """
        Compute the average precision (or a different metric) for the top n
        recommendations.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        top_n_ranks_with_labels = LanguageModelScorer.display_top_n(
            language_model_data, feature_weights, n
        ).pipe(LanguageModelScorer.add_labels, language_model_data.integer_labels)

        return metric.from_df(top_n_ranks_with_labels)
