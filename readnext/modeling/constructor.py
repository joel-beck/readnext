from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl

from readnext.modeling.constructor_plugin import ModelDataConstructorPlugin
from readnext.modeling.document_info import DocumentInfo
from readnext.utils import (
    CitationFeaturesFrame,
    InfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
)


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
    constructor_plugin: ModelDataConstructorPlugin
    info_columns: list[str] = field(
        default_factory=lambda: ["candidate_d3_document_id", "title", "author", "arxiv_labels"]
    )
    feature_columns: list[str]

    query_document: DocumentInfo = field(init=False)

    def __post_init__(self) -> None:
        """Store the query document information in an instance attribute during initialization."""
        self.query_document = self.constructor_plugin.collect_query_document()

    @abstractmethod
    def get_features_frame(self) -> CitationFeaturesFrame | LanguageFeaturesFrame:
        """
        Construct the feature matrix consisting of a `candidate_d3_document_id` column
        and all rank columns.
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

    def get_info_frame(self) -> InfoFrame:
        """
        Exclude the query document from the documents data and select only the feature
        columns with information about the candidate documents.
        """
        return self.get_query_documents_data().select(self.info_columns)

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

    def get_integer_labels_frame(self) -> IntegerLabelsFrame:
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
