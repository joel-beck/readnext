from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from readnext.modeling.document_info import DocumentInfo, DocumentScore


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

    query_document_id: int
    documents_data: pd.DataFrame
    info_cols: list[str]
    query_document: DocumentInfo = field(init=False)

    def __post_init__(self) -> None:
        """Store the query document information in an instance attribute during initialization."""
        self.query_document = self.collect_query_document()

    @abstractmethod
    def extend_info_matrix(self, info_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Add additonal features to the information matrix which contains the candidate
        document features.
        """

    def collect_query_document(self) -> DocumentInfo:
        """Extract and collect the query document information from the documents data."""
        query_document_title = str(self.documents_data.loc[self.query_document_id, "title"])
        query_document_author = str(self.documents_data.loc[self.query_document_id, "author"])
        query_document_labels = cast(
            list[str], self.documents_data.loc[self.query_document_id, "arxiv_labels"]
        )

        return DocumentInfo(
            document_id=self.query_document_id,
            title=query_document_title,
            author=query_document_author,
            arxiv_labels=query_document_labels,
        )

    def exclude_query_document(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exclude the query document from the documents data."""
        return df.loc[df.index != self.query_document_id]

    def filter_documents_data(self) -> pd.DataFrame:
        """
        Exclude the query document from the documents data and return the filtered data.
        """
        return self.exclude_query_document(self.documents_data)

    def get_info_matrix(self) -> pd.DataFrame:
        """
        Exclude the query document from the documents data and select only the feature
        columns with information about the candidate documents.
        """
        return self.filter_documents_data().loc[:, self.info_cols]

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

    def get_integer_labels(self) -> pd.Series:
        """
        Extract the arxiv labels for all candidate documents from the input data and
        convert them to integer labels.
        """
        return (
            self.filter_documents_data()
            .loc[:, "arxiv_labels"]
            .apply(self.shares_arxiv_label)
            .apply(self.boolean_to_int)
            .rename("integer_labels")
        )

    def document_scores_to_frame(self, document_scores: list[DocumentScore]) -> pd.DataFrame:
        """
        Convert the scores of all candidate documents to a dataframe. The output
        dataframe has a single `score` column with the document ids as index.
        """
        return pd.DataFrame(
            [
                {
                    "document_id": document_score.document_info.document_id,
                    "score": document_score.score,
                }
                for document_score in document_scores
            ]
        ).set_index("document_id")


@dataclass(kw_only=True)
class CitationModelDataConstructor(ModelDataConstructor):
    """
    Constructor for the citation-based recommender model data. Takes the co-citation
    analysis and bibliographic coupling scores as additional inputs.
    """

    co_citation_analysis_scores: pd.DataFrame
    bibliographic_coupling_scores: pd.DataFrame
    info_cols: list[str] = field(
        default_factory=lambda: [
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
            "publication_date_rank",
            "citationcount_document_rank",
            "citationcount_author_rank",
        ]
    )

    def get_citation_method_scores(self, citation_method_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the scores of all candidate documents for a given citation method and
        converts them to a dataframe with a single `score` column and the document ids
        as index.
        """
        document_scores: list[DocumentScore] = citation_method_data.loc[
            self.query_document_id
        ].item()

        return self.document_scores_to_frame(document_scores)

    def get_co_citation_analysis_scores(self) -> pd.DataFrame:
        """
        Extract the co-citation analysis scores of all candidate documents and converts
        them to a dataframe with a single `co_citation_analysis` column and the document
        ids as index.
        """
        return self.get_citation_method_scores(self.co_citation_analysis_scores).rename(
            columns={"score": "co_citation_analysis"}
        )

    def get_bibliographic_coupling_scores(self) -> pd.DataFrame:
        """
        Extract the bibliographic coupling scores of all candidate documents and
        converts them to a dataframe with a single `bibliographic_coupling` column and
        the document ids as index.
        """
        return self.get_citation_method_scores(self.bibliographic_coupling_scores).rename(
            columns={"score": "bibliographic_coupling"}
        )

    def extend_info_matrix(self, info_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the co-citation analysis and bibliographic coupling scores to the
        information features about the candidate documents.
        """
        return info_matrix.assign(
            co_citation_analysis=self.get_co_citation_analysis_scores(),
            bibliographic_coupling=self.get_bibliographic_coupling_scores(),
        )

    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Collects all citation-based and global document feature ranks that are used for
        the weighted citation recommender model in a single dataframe.
        """
        return (
            self.filter_documents_data()
            .loc[:, self.feature_cols]
            .assign(
                co_citation_analysis_rank=self.get_co_citation_analysis_scores().rank(
                    ascending=False
                ),
                bibliographic_coupling_rank=self.get_bibliographic_coupling_scores().rank(
                    ascending=False
                ),
            )
        )


@dataclass(kw_only=True)
class LanguageModelDataConstructor(ModelDataConstructor):
    """
    Constructor for the language-based recommender model data. Takes the cosine
    similarities of all candidate documents with respect to the query document as
    additional input.
    """

    cosine_similarities: pd.DataFrame
    info_cols: list[str] = field(default_factory=lambda: ["title", "author", "arxiv_labels"])

    def get_cosine_similarity_scores(self) -> pd.DataFrame:
        """
        Extracts the cosine similarity scores of all candidate documents with respect to
        the query document and converts them to a dataframe with a single
        `cosine_similarity` column and the document ids as index.
        """
        document_scores: list[DocumentScore] = self.cosine_similarities.loc[
            self.query_document_id
        ].item()

        return self.document_scores_to_frame(document_scores).rename(
            columns={"score": "cosine_similarity"}
        )

    def extend_info_matrix(self, info_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the cosine similarity scores to the information features about the
        candidate documents.
        """
        return info_matrix.assign(
            cosine_similarity=self.get_cosine_similarity_scores(),
        )

    def get_cosine_similarity_ranks(self) -> pd.DataFrame:
        """
        Computes the cosine similarity ranks of all candidate documents with respect to
        the query document. The output dataframe has a single `cosine_similarity_rank`
        column and the document ids as index.
        """
        return (
            self.get_cosine_similarity_scores()
            .rank(ascending=False)
            .rename({"cosine_similarity": "cosine_similarity_rank"}, axis="columns")
        )
