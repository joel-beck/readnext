from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import polars as pl

from readnext.config import MagicNumbers
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
        n: int = MagicNumbers.n_recommendations,
    ) -> float:
        ...

    @overload
    @staticmethod
    @abstractmethod
    def score_top_n(
        model_data: TModelData,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> int:
        ...

    @staticmethod
    @abstractmethod
    def score_top_n(
        model_data: TModelData,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float | int:
        ...

    @staticmethod
    @abstractmethod
    def display_top_n(
        model_data: TModelData,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> pl.DataFrame:
        ...

    @staticmethod
    def merge_on_candidate_d3_document_id(df: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
        """Merge two dataframes on the `candidate_d3_document_id` column."""
        return df.join(other, on="candidate_d3_document_id", how="left")

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
        Select and collect the top n recommendations from a citation model in a
        dataframe.

        The output dataframe has two columns named `candidate_d3_document_id` and
        `weighted_rank`
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
    def display_top_n(
        citation_model_data: CitationModelData,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a citation model in a
        dataframe. Add rank columns of the feature matrix and raw score columns of the
        info matrix to the output dataframe.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        return (
            CitationModelScorer.select_top_n_ranks(citation_model_data, feature_weights, n)
            .pipe(
                CitationModelScorer.merge_on_candidate_d3_document_id,
                citation_model_data.feature_matrix,
            )
            .pipe(
                CitationModelScorer.merge_on_candidate_d3_document_id,
                citation_model_data.info_matrix,
            )
        )

    @overload
    @staticmethod
    def score_top_n(
        citation_model_data: CitationModelData,
        metric: AveragePrecision,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float:
        ...

    @overload
    @staticmethod
    def score_top_n(
        citation_model_data: CitationModelData,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> int:
        ...

    @staticmethod
    def score_top_n(
        citation_model_data: CitationModelData,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float | int:
        """
        Compute the average precision (or a different metric) for the top n
        recommendations.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        top_n_ranks_with_labels = CitationModelScorer.display_top_n(
            citation_model_data, feature_weights, n
        ).pipe(
            CitationModelScorer.merge_on_candidate_d3_document_id,
            citation_model_data.integer_labels,
        )

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
        Select and collect the top n recommendations from a language model in a
        dataframe.

        The output dataframe has two columns named `candidate_d3_document_id` and
        `cosine_similarity_rank`.

        The `feature_weights` argument is not used for scoring language models.
        """
        return language_model_data.cosine_similarity_ranks.sort("cosine_similarity_rank").head(n)

    @staticmethod
    def move_cosine_similarity_second(df: pl.DataFrame) -> pl.DataFrame:
        """
        Move the `cosine_similarity` column with cosine similarity scores to the second
        position after the `candidate_d3_document_id` column in a dataframe.
        """
        return df.select(
            [
                "candidate_d3_document_id",
                "cosine_similarity",
                pl.exclude(["candidate_d3_document_id", "cosine_similarity"]),
            ]
        )

    @staticmethod
    def display_top_n(
        language_model_data: LanguageModelData,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a citation model in a
        dataframe. Add cosine similarity ranks and raw score columns of the info matrix
        to the output dataframe.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        return (
            LanguageModelScorer.select_top_n_ranks(language_model_data, feature_weights, n)
            .pipe(
                LanguageModelScorer.merge_on_candidate_d3_document_id,
                language_model_data.info_matrix,
            )
            .pipe(LanguageModelScorer.move_cosine_similarity_second)
            # ranks are redundant information given the raw cosine similarity scores in
            # sorted order
            .drop("cosine_similarity_rank")
        )

    @overload
    @staticmethod
    def score_top_n(
        language_model_data: LanguageModelData,
        metric: AveragePrecision,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float:
        ...

    @overload
    @staticmethod
    def score_top_n(
        language_model_data: LanguageModelData,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> int:
        ...

    @staticmethod
    def score_top_n(
        language_model_data: LanguageModelData,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float | int:
        """
        Compute the average precision (or a different metric) for the top n
        recommendations.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        top_n_ranks_with_labels = LanguageModelScorer.display_top_n(
            language_model_data, feature_weights, n
        ).pipe(
            LanguageModelScorer.merge_on_candidate_d3_document_id,
            language_model_data.integer_labels,
        )

        return metric.from_df(top_n_ranks_with_labels)
