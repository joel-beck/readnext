from dataclasses import dataclass, field

import polars as pl

from readnext.config import MagicNumbers
from readnext.modeling.constructor import ModelDataConstructor
from readnext.utils.aliases import (
    CandidateScoresFrame,
    CitationFeaturesFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    ScoresFrame,
)
from readnext.utils.repr import generate_frame_repr


@dataclass(kw_only=True)
class CitationModelDataConstructor(ModelDataConstructor):
    """
    Constructor for the citation-based recommender model data. Takes the co-citation
    analysis and bibliographic coupling scores as additional inputs.
    """

    co_citation_analysis_scores_frame: ScoresFrame | CandidateScoresFrame
    bibliographic_coupling_scores_frame: ScoresFrame | CandidateScoresFrame
    info_columns: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "title",
            "author",
            "arxiv_labels",
            "semanticscholar_url",
            "arxiv_url",
        ]
    )
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "publication_date",
            "citationcount_document",
            "citationcount_author",
            "co_citation_analysis_score",
            "bibliographic_coupling_score",
        ]
    )
    rank_columns: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "publication_date_rank",
            "citationcount_document_rank",
            "citationcount_author_rank",
            "co_citation_analysis_rank",
            "bibliographic_coupling_rank",
        ]
    )
    points_columns: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "publication_date_points",
            "citationcount_document_points",
            "citationcount_author_points",
            "co_citation_analysis_points",
            "bibliographic_coupling_points",
        ]
    )

    def __repr__(self) -> str:
        d3_document_id_repr = f"d3_document_id={self.d3_document_id}"
        query_document_repr = f"query_document={self.query_document!r}"
        documents_frame_repr = f"documents_frame={generate_frame_repr(self.documents_frame)}"
        constructor_plugin_repr = f"constructor_plugin={self.constructor_plugin!r}"
        co_citation_analysis_scores_frame_repr = (
            f"co_citation_analysis_scores_frame="
            f"{generate_frame_repr(self.co_citation_analysis_scores_frame)}"
        )
        bibliographic_coupling_scores_frame_repr = (
            f"bibliographic_coupling_scores_frame="
            f"{generate_frame_repr(self.bibliographic_coupling_scores_frame)}"
        )
        info_columns_repr = f"info_columns={self.info_columns}"
        feature_columns_repr = f"feature_columns={self.feature_columns}"
        rank_columns_repr = f"rank_columns={self.rank_columns}"
        points_columns_repr = f"points_columns={self.points_columns}"

        return (
            f"{self.__class__.__name__}(\n"
            f"  {d3_document_id_repr},\n"
            f"  {query_document_repr},\n"
            f"  {documents_frame_repr},\n"
            f"  {constructor_plugin_repr},\n"
            f"  {co_citation_analysis_scores_frame_repr},\n"
            f"  {bibliographic_coupling_scores_frame_repr},\n"
            f"  {info_columns_repr},\n"
            f"  {feature_columns_repr},\n"
            f"  {rank_columns_repr},\n"
            f"  {points_columns_repr}\n"
            ")"
        )

    def get_co_citation_analysis_scores(self) -> CandidateScoresFrame:
        """
        Extract the co-citation analysis scores of all candidate documents and converts
        them to a dataframe with with two columns named `candidate_d3_document_id`
        and `score`.
        """
        return self.constructor_plugin.get_candidate_scores(self.co_citation_analysis_scores_frame)

    def get_bibliographic_coupling_scores(self) -> CandidateScoresFrame:
        """
        Extract the bibliographic coupling scores of all candidate documents and
        converts them to a dataframe with with two columns named `candidate_d3_document_id`
        and `score`.
        """
        return self.constructor_plugin.get_candidate_scores(
            self.bibliographic_coupling_scores_frame
        )

    def get_features_frame(self) -> CitationFeaturesFrame:
        """
        Collects all citation feature into a dataframe with six columns:
        `candidate_d3_document_id`, `publication_date`, `citationcount_document`,
        `citationcount_author`, `co_citation_analysis_score`, and
        `bibliographic_coupling_score`.

        Each joined scores frame contains a subset of the rows of the query documents
        frame. For weighted scoring all candidate documents are kept (left join) and the
        score is set to 0 for documents not present in the scores frame.
        """
        return (
            self.get_query_documents_frame()
            .join(
                self.get_co_citation_analysis_scores(),
                on="candidate_d3_document_id",
                how="left",
            )
            .rename({"score": "co_citation_analysis_score"})
            .join(
                self.get_bibliographic_coupling_scores(),
                on="candidate_d3_document_id",
                how="left",
            )
            .rename({"score": "bibliographic_coupling_score"})
            .select(self.feature_columns)
            .fill_null(0)
        )

    @staticmethod
    def get_ranks_from_scores_column(expression: pl.Expr) -> pl.Expr:
        """
        Computes ranks from a given scores column. In case of ties the minimum rank is
        used.
        Example: scores = [1, 2, 2, 3] -> ranks = [4, 2, 2, 1]

        Returns an expression that can be
        used to create a new column in a dataframe.
        """
        return expression.rank(descending=True, method="min").cast(pl.Float32)

    @staticmethod
    def threshold_rank_column(expression: pl.Expr) -> pl.Expr:
        """
        Set all ranks above a given threshold to the maximum rank + 1.
        """
        return (
            pl.when(expression > MagicNumbers.scoring_limit)
            .then(MagicNumbers.scoring_limit + 1)
            .otherwise(expression)
        )

    def get_ranks_frame(self, features_frame: CitationFeaturesFrame) -> CitationRanksFrame:
        """
        Collects all citation-based and global document feature ranks that are used for
        the weighted citation recommender model in a single dataframe.

        The output dataframe has six columns named `candidate_d3_document_id`,
        `publication_date_rank`, `citationcount_document_rank`,
        `citationcount_author_rank`, `co_citation_analysis_rank`, and
        `bibliographic_coupling_rank`.
        """
        return features_frame.with_columns(
            publication_date_rank=pl.col("publication_date")
            .pipe(self.get_ranks_from_scores_column)
            .pipe(self.threshold_rank_column),
            citationcount_document_rank=pl.col("citationcount_document")
            .pipe(self.get_ranks_from_scores_column)
            .pipe(self.threshold_rank_column),
            citationcount_author_rank=pl.col("citationcount_author")
            .pipe(self.get_ranks_from_scores_column)
            .pipe(self.threshold_rank_column),
            co_citation_analysis_rank=pl.col("co_citation_analysis_score")
            .pipe(self.get_ranks_from_scores_column)
            .pipe(self.threshold_rank_column),
            bibliographic_coupling_rank=pl.col("bibliographic_coupling_score")
            .pipe(self.get_ranks_from_scores_column)
            .pipe(self.threshold_rank_column),
        ).select(self.rank_columns)

    @staticmethod
    def get_points_from_rank_column(expression: pl.Expr) -> pl.Expr:
        """
        Compute the points of a feature for a given feature rank column. The points are
        used to establish the weighted ranking, documents with a higher weighted point
        score are ranked higher.

        Rank 1 obtains the maximum number of points (the scoring limit value + 1), the
        rank of the scoring limit obtains 1 point, and all ranks above the scoring limit
        obtain 0 points.
        """
        return ((MagicNumbers.scoring_limit + 1) - expression).cast(pl.Float32)

    def get_points_frame(self, ranks_frame: CitationRanksFrame) -> CitationPointsFrame:
        """
        Collects all citation-based and global document feature points that are used for
        the weighted citation recommender model in a single dataframe.

        The output dataframe has six columns named `candidate_d3_document_id`,
        `publication_date_points`, `citationcount_document_points`,
        `citationcount_author_points`, `co_citation_analysis_points`, and
        `bibliographic_coupling_points`.
        """
        return ranks_frame.with_columns(
            publication_date_points=pl.col("publication_date_rank").pipe(
                self.get_points_from_rank_column
            ),
            citationcount_document_points=pl.col("citationcount_document_rank").pipe(
                self.get_points_from_rank_column
            ),
            citationcount_author_points=pl.col("citationcount_author_rank").pipe(
                self.get_points_from_rank_column
            ),
            co_citation_analysis_points=pl.col("co_citation_analysis_rank").pipe(
                self.get_points_from_rank_column
            ),
            bibliographic_coupling_points=pl.col("bibliographic_coupling_rank").pipe(
                self.get_points_from_rank_column
            ),
        ).select(self.points_columns)
