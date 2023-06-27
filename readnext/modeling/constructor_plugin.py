from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl

from readnext.data import SemanticScholarResponse
from readnext.modeling.document_info import DocumentInfo
from readnext.utils.aliases import CandidateScoresFrame, DocumentsFrame, ScoresFrame
from readnext.utils.repr import generate_frame_repr


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
    def get_candidate_scores(
        self, scores_frame: ScoresFrame | CandidateScoresFrame
    ) -> CandidateScoresFrame:
        ...


@dataclass
class SeenModelDataConstructorPlugin(ModelDataConstructorPlugin):
    """
    `ModelDataConstructor` methods for seen query documents only.
    """

    d3_document_id: int
    documents_frame: DocumentsFrame

    def __repr__(self) -> str:
        d3_document_id_repr = f"d3_document_id={self.d3_document_id}"
        documents_frame_repr = f"documents_frame={generate_frame_repr(self.documents_frame)}"

        return (
            f"{self.__class__.__name__}(\n"
            f"  {d3_document_id_repr},\n"
            f"  {documents_frame_repr}\n"
            ")"
        )

    def collect_query_document(self) -> DocumentInfo:
        """Extract and collect the query document information from the documents data."""
        query_document_row = self.documents_frame.filter(
            pl.col("d3_document_id") == self.d3_document_id
        )

        return DocumentInfo(
            d3_document_id=self.d3_document_id,
            title=query_document_row.select("title").item(),
            author=query_document_row.select("author").item(),
            publication_date=query_document_row.select("publication_date").item(),
            arxiv_labels=query_document_row.select("arxiv_labels").item().to_list(),
            semanticscholar_url=query_document_row.select("semanticscholar_url").item(),
            arxiv_url=query_document_row.select("arxiv_url").item(),
            abstract=query_document_row.select("abstract").item(),
        )

    def get_candidate_scores(self, scores_frame: ScoresFrame) -> CandidateScoresFrame:
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
        return DocumentInfo(
            d3_document_id=-1,
            title=self.response.title,
            semanticscholar_url=self.response.semanticscholar_url,
            arxiv_url=self.response.arxiv_url,
            abstract=self.response.abstract,
        )

    def get_candidate_scores(self, scores_frame: CandidateScoresFrame) -> CandidateScoresFrame:
        """
        For unseen documents the query document is not contained in the training data.
        Thus, a filtering step is not necessary. The input and output dataframes contain
        only the two columns `candidate_d3_document_id` and `score`.
        """

        return scores_frame
