from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import polars as pl

from readnext.config import MagicNumbers
from readnext.evaluation.metrics import AveragePrecision, CountUniqueLabels
from readnext.modeling import CitationModelData, LanguageModelData, ModelData
from readnext.utils import CitationPointsFrame

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

    model_data: TModelData

    @abstractmethod
    def select_top_n(self, feature_weights: FeatureWeights, n: int) -> pl.DataFrame:
        ...

    @abstractmethod
    def display_top_n(
        self,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> pl.DataFrame:
        ...

    @overload
    @abstractmethod
    def score_top_n(
        self,
        metric: AveragePrecision,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float:
        ...

    @overload
    @abstractmethod
    def score_top_n(
        self,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> int:
        ...

    @abstractmethod
    def score_top_n(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float | int:
        ...

    @staticmethod
    def merge_on_candidate_d3_document_id(df: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
        """
        Merge two dataframes on the `candidate_d3_document_id` column. Output dataframe
        contains only the rows present in both input dataframes (inner join).
        """
        return df.join(other, on="candidate_d3_document_id", how="inner")


@dataclass
class CitationModelScorer(ModelScorer):
    model_data: CitationModelData

    @staticmethod
    def compute_weighted_points(
        points_frame: CitationPointsFrame, feature_weights: FeatureWeights
    ) -> pl.DataFrame:
        """
        Compute the weighted rowsums of a dataframe with one weight for each citation
        feature column. The output dataframe has two columns named
        `candidate_d3_document_id` and `weighted_points`.
        """

        return points_frame.with_columns(
            weighted_points=(
                feature_weights.publication_date * points_frame["publication_date_points"]
                + feature_weights.citationcount_document
                * points_frame["citationcount_document_points"]
                + feature_weights.citationcount_author * points_frame["citationcount_author_points"]
                + feature_weights.co_citation_analysis * points_frame["co_citation_analysis_points"]
                + feature_weights.bibliographic_coupling
                * points_frame["bibliographic_coupling_points"]
            )
        ).select(["candidate_d3_document_id", "weighted_points"])

    def select_top_n(self, feature_weights: FeatureWeights, n: int) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a citation model in a
        dataframe sorted in descending order by the `weighted_points` column.

        The output dataframe has two columns named `candidate_d3_document_id` and
        `weighted_rank`
        """

        return (
            self.compute_weighted_points(self.model_data.points_frame, feature_weights)
            .sort("weighted_points", descending=True)
            .head(n)
        )

    def display_top_n(
        self, feature_weights: FeatureWeights | None = None, n: int = MagicNumbers.n_recommendations
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a citation model in a
        dataframe.

        Note that the row order is NOT maintained by left joins! Thus, the output
        dataframe has to be sorted again in descending order by the `weighted_points`
        column.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        return (
            self.select_top_n(feature_weights, n)
            # merge points frame first to position individual feature points next to
            # weighted points, columns are added from left to right
            .pipe(
                self.merge_on_candidate_d3_document_id,
                self.model_data.points_frame,
            )
            .pipe(
                self.merge_on_candidate_d3_document_id,
                self.model_data.info_frame,
            )
            .pipe(self.merge_on_candidate_d3_document_id, self.model_data.integer_labels_frame)
            .pipe(
                self.merge_on_candidate_d3_document_id,
                self.model_data.features_frame,
            )
            .sort("weighted_points", descending=True)
        )

    @overload
    def score_top_n(
        self,
        metric: AveragePrecision,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float:
        ...

    @overload
    def score_top_n(
        self,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> int:
        ...

    def score_top_n(
        self,
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

        # the scoring metric only requires the `integer_labels` column in
        # `integer_labels_frame`, but calling `display_top_n` first is necessary to
        # select the top n recommendations and determine the recommendation order!
        top_n = self.display_top_n(feature_weights, n)
        return metric.from_df(top_n)


@dataclass
class LanguageModelScorer(ModelScorer):
    model_data: LanguageModelData

    def select_top_n(
        self,
        feature_weights: FeatureWeights,  # noqa: ARG002
        n: int,
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a language model in a
        dataframe.

        The output dataframe has two columns named `candidate_d3_document_id` and
        `cosine_similarity_rank`.

        The `feature_weights` argument is not used for scoring language models.
        """
        return self.model_data.features_frame.sort("cosine_similarity", descending=True).head(n)

    def display_top_n(
        self,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> pl.DataFrame:
        """
        Select and collect the top n recommendations from a citation model in a
        dataframe based on cosine similarity.

        Note that the row order is NOT maintained by left joins! Thus, the output
        dataframe has to be sorted again in descending order by the `cosine_similarity`
        column.
        """
        if feature_weights is None:
            feature_weights = FeatureWeights()

        return (
            self.select_top_n(feature_weights, n)
            .pipe(
                self.merge_on_candidate_d3_document_id,
                self.model_data.info_frame,
            )
            .pipe(self.merge_on_candidate_d3_document_id, self.model_data.integer_labels_frame)
            .sort("cosine_similarity", descending=True)
        )

    @overload
    def score_top_n(
        self,
        metric: AveragePrecision,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> float:
        ...

    @overload
    def score_top_n(
        self,
        metric: CountUniqueLabels,
        feature_weights: FeatureWeights | None = None,
        n: int = MagicNumbers.n_recommendations,
    ) -> int:
        ...

    def score_top_n(
        self,
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

        top_n = self.display_top_n(feature_weights, n)
        return metric.from_df(top_n)
