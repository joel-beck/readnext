from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl

from readnext.modeling.constructor_plugin import ModelDataConstructorPlugin
from readnext.modeling.document_info import DocumentInfo
from readnext.utils import CandidateRanksFrame, CandidateScoresFrame, ScoresFrame


@dataclass(kw_only=True)
class ModelDataConstructor(ABC):
    """
    Intermediate object that bridges the gap between the input documents data and the
    required model data.

    Extracts features for a single query document from the documents data and serves as
    constructor for the `ModelData` object that is used for fitting the recommender
    model. Takes the document id of the query document, the input documents data, and a
    list of feature names with information about the candidate documents as input.
    """

    d3_document_id: int
    documents_data: pl.DataFrame
    info_cols: list[str]
    constructor_plugin: ModelDataConstructorPlugin

    query_document: DocumentInfo = field(init=False)

    def __post_init__(self) -> None:
        """Store the query document information in an instance attribute during initialization."""
        self.query_document = self.constructor_plugin.collect_query_document()

    @abstractmethod
    def extend_info_matrix(self, info_matrix: pl.DataFrame) -> pl.DataFrame:
        """
        Add additonal features to the information matrix which contains the candidate
        document features.
        """

    def exclude_query_document(self) -> pl.DataFrame:
        """Exclude the query document from the documents data."""
        return self.documents_data.filter(pl.col("d3_document_id") != self.d3_document_id)

    @staticmethod
    def rename_to_candidate_id(df: pl.DataFrame) -> pl.DataFrame:
        """
        Once the query document is excluded from the Dataframe, rename the
        `d3_document_id` columns to `candidate_d3_document_id` since all remaining
        documents are candidate documents.
        """
        return df.rename({"d3_document_id": "candidate_d3_document_id"})

    def get_query_documents_data(self) -> pl.DataFrame:
        """
        Get a subset of the documents data for a single query document by filtering out
        the query document row and renaming the `d3_document_id` column to
        `candidate_d3_document_id`.
        """
        return self.exclude_query_document().pipe(self.rename_to_candidate_id)

    def get_info_matrix(self) -> pl.DataFrame:
        """
        Exclude the query document from the documents data and select only the feature
        columns with information about the candidate documents.
        """
        return self.get_query_documents_data().select(self.info_cols)

    def get_candidate_ranks(
        self, scores_frame: ScoresFrame | CandidateScoresFrame
    ) -> CandidateRanksFrame:
        """
        Computes ranks of all candidate documents for a given scores frame from the
        corresponding scores. Returns a dataframe with two columns named
        `candidate_d3_document_id` and `rank`.
        """
        return (
            self.constructor_plugin.get_candidate_scores(scores_frame)
            .with_columns(rank=pl.col("score").rank(descending=True))
            .drop("score")
        )

    def shares_arxiv_label(
        self,
        candidate_document_labels: list[str],
    ) -> bool:
        """
        Examine if a candidate document shares at least one arxiv label with the query
        document.
        """
        return any(label in candidate_document_labels for label in self.query_document.arxiv_labels)

    @staticmethod
    def boolean_to_int(boolean: bool) -> int:
        """Convert a boolean value to a 0/1 integer."""
        return int(boolean)

    def get_integer_labels(self) -> pl.DataFrame:
        """
        Extract the arxiv labels for all candidate documents from the input data and
        convert them to integer labels. Return a dataframe with two columns named
        `candidate_d3_document_id` and `integer_labels`.
        """
        return (
            self.get_query_documents_data()
            .select(["candidate_d3_document_id", "arxiv_labels"])
            .with_columns(
                integer_labels=pl.col("arxiv_labels")
                .apply(self.shares_arxiv_label)
                .apply(self.boolean_to_int)
            )
            .drop("arxiv_labels")
        )


@dataclass(kw_only=True)
class CitationModelDataConstructor(ModelDataConstructor):
    """
    Constructor for the citation-based recommender model data. Takes the co-citation
    analysis and bibliographic coupling scores as additional inputs.
    """

    co_citation_analysis_scores: ScoresFrame | CandidateScoresFrame
    bibliographic_coupling_scores: ScoresFrame | CandidateScoresFrame
    info_cols: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "title",
            "author",
            "arxiv_labels",
            "publication_date",
            "citationcount_document",
            "citationcount_author",
        ]
    )
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "publication_date_rank",
            "citationcount_document_rank",
            "citationcount_author_rank",
        ]
    )

    def get_co_citation_analysis_scores(self) -> CandidateScoresFrame:
        """
        Extract the co-citation analysis scores of all candidate documents and converts
        them to a dataframe with with two columns named `candidate_d3_document_id`
        and `score`.
        """
        return self.constructor_plugin.get_candidate_scores(self.co_citation_analysis_scores)

    def get_bibliographic_coupling_scores(self) -> CandidateScoresFrame:
        """
        Extract the bibliographic coupling scores of all candidate documents and
        converts them to a dataframe with with two columns named `candidate_d3_document_id`
        and `score`.
        """
        return self.constructor_plugin.get_candidate_scores(self.bibliographic_coupling_scores)

    def get_co_citation_analysis_ranks(self) -> CandidateRanksFrame:
        """
        Extract the co-citation analysis ranks of all candidate documents and converts
        them to a dataframe with with two columns named `candidate_d3_document_id`
        and `rank`.
        """
        return self.get_candidate_ranks(self.co_citation_analysis_scores)

    def get_bibliographic_coupling_ranks(self) -> CandidateRanksFrame:
        """
        Extract the bibliographic coupling ranks of all candidate documents and
        converts them to a dataframe with with two columns named `candidate_d3_document_id`
        and `rank`.
        """
        return self.get_candidate_ranks(self.bibliographic_coupling_scores)

    def extend_info_matrix(self, info_matrix: pl.DataFrame) -> pl.DataFrame:
        """
        Adds the co-citation analysis and bibliographic coupling scores to the
        information features about the candidate documents. Join them sequentially on
        the shared `candidate_d3_document_id` column.

        Renames the `score` columns of the co-citation analysis and bibliographic candidate score
        frames to more specific names.
        """
        return (
            info_matrix.join(
                self.get_co_citation_analysis_scores(), on="candidate_d3_document_id", how="left"
            )
            .rename({"score": "co_citation_analysis"})
            .join(
                self.get_bibliographic_coupling_scores(), on="candidate_d3_document_id", how="left"
            )
            .rename({"score": "bibliographic_coupling"})
        )

    def get_feature_matrix(self) -> pl.DataFrame:
        """
        Collects all citation-based and global document feature ranks that are used for
        the weighted citation recommender model in a single dataframe.

        Renames the `rank` columns of the co-citation analysis and bibliographic
        candidate rank frames to more specific names.
        """
        return (
            self.get_query_documents_data()
            .select(self.feature_cols)
            .join(self.get_co_citation_analysis_ranks(), on="candidate_d3_document_id", how="left")
            .rename({"rank": "co_citation_analysis_rank"})
            .join(
                self.get_bibliographic_coupling_ranks(), on="candidate_d3_document_id", how="left"
            )
            .rename({"rank": "bibliographic_coupling_rank"})
        )


@dataclass(kw_only=True)
class LanguageModelDataConstructor(ModelDataConstructor):
    """
    Constructor for the language-based recommender model data. Takes the cosine
    similarities of all candidate documents with respect to the query document as
    additional input.
    """

    cosine_similarities: ScoresFrame | CandidateScoresFrame
    info_cols: list[str] = field(
        default_factory=lambda: ["candidate_d3_document_id", "title", "author", "arxiv_labels"]
    )

    def get_cosine_similarity_scores(self) -> CandidateScoresFrame:
        """
        Extracts the cosine similarity scores of all candidate documents with respect to
        the query document and converts them to a dataframe with two columns named
        `candidate_d3_document_id` and `score`.
        """
        # output dataframe has length of original full data in tests, even though the
        # test_cosine_similarities data itself only contains 100 rows
        return self.constructor_plugin.get_candidate_scores(self.cosine_similarities)

    def get_cosine_similarity_ranks(self) -> CandidateRanksFrame:
        """
        Computes the cosine similarity ranks of all candidate documents with respect to
        the query document.

        Renames the `rank` column of the cosine similarity candidate rank frame to a
        more specific name.
        """
        return self.get_candidate_ranks(self.cosine_similarities).rename(
            {"rank": "cosine_similarity_rank"}
        )

    def extend_info_matrix(self, info_matrix: pl.DataFrame) -> pl.DataFrame:
        """
        Adds the cosine similarity scores to the information features about the
        candidate documents.

        Renames the `score` column of the cosine similarity candidate score frame to a
        more specific name.
        """
        return info_matrix.join(
            self.get_cosine_similarity_scores(), on="candidate_d3_document_id", how="left"
        ).rename({"score": "cosine_similarity"})
