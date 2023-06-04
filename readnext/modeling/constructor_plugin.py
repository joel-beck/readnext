from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl

from readnext.data import SemanticScholarResponse
from readnext.modeling.document_info import DocumentInfo


@dataclass
class ModelDataConstructorPlugin(ABC):
    """
    Provides methods for the `ModelDataConstructor` class that are different for seen
    and unseen papers.
    """

    @abstractmethod
    def collect_query_document(self) -> DocumentInfo:
        ...

    @abstractmethod
    def get_query_scores(self, candidate_scores_frame: pl.DataFrame) -> pl.DataFrame:
        ...


@dataclass
class SeenModelDataConstructorPlugin(ModelDataConstructorPlugin):
    """
    `ModelDataConstructor` methods for seen query documents only.
    """

    d3_document_id: int
    documents_data: pl.DataFrame

    def collect_query_document(self) -> DocumentInfo:
        """Extract and collect the query document information from the documents data."""
        query_document_row = self.documents_data.filter(
            pl.col("d3_document_id") == self.d3_document_id
        )

        return DocumentInfo(
            d3_document_id=self.d3_document_id,
            title=query_document_row.select("title").item(),
            author=query_document_row.select("author").item(),
            arxiv_labels=query_document_row.select("arxiv_labels").item().to_list(),
            abstract=query_document_row.select("abstract").item(),
        )

    def get_query_scores(self, scores_frame: pl.DataFrame) -> pl.DataFrame:
        """
        Extract the scores of all candidate documents for a given scores frame and
        converts them to a dataframe with two columns named `candidate_d3_document_id`
        and `score`.
        """
        return scores_frame.filter(pl.col("query_d3_document_id") == self.d3_document_id).drop(
            "query_d3_document_id"
        )


@dataclass
class UnseenModelDataConstructorPlugin(ModelDataConstructorPlugin):
    """
    `ModelDataConstructor` methods for unseen query documents only.
    """

    response: SemanticScholarResponse

    def collect_query_document(self) -> DocumentInfo:
        title = self.response.title if self.response.title is not None else ""
        abstract = self.response.abstract if self.response.abstract is not None else ""

        return DocumentInfo(d3_document_id=-1, title=title, abstract=abstract)

    def get_query_scores(self, candidate_scores_frame: pl.DataFrame) -> pl.DataFrame:
        """
        For unseen documents the query document is not contained in the training data.
        Thus, a filtering step is not necessary. The input and output dataframes contain
        only the two columns `candidate_d3_document_id` and `score`.
        """

        return candidate_scores_frame
