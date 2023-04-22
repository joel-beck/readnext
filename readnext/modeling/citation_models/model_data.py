from dataclasses import dataclass, field
from typing import cast

import pandas as pd


@dataclass
class Document:
    document_id: int
    title: str
    author: str
    labels: list[str]


@dataclass
class CitationModelData:
    input_document: Document
    info_matrix: pd.DataFrame
    feature_matrix: pd.DataFrame
    labels: pd.Series


@dataclass
class CitationModelDataFromId:
    document_id: int
    documents_data: pd.DataFrame
    co_citation_analysis_data: pd.DataFrame
    bibliographic_coupling_data: pd.DataFrame
    info_cols: list[str] = field(
        default_factory=lambda: [
            "title",
            "author",
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
    input_document: Document = field(init=False)

    def __post_init__(self) -> None:
        input_document_title = str(self.documents_data.loc[self.document_id, "title"])
        input_document_author = str(self.documents_data.loc[self.document_id, "author"])
        input_document_labels = cast(
            list[str], self.documents_data.loc[self.document_id, "arxiv_labels"]
        )

        self.input_document = Document(
            self.document_id, input_document_title, input_document_author, input_document_labels
        )

    def exclude_input_document(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df.index != self.document_id]

    def filter_documents_data(self) -> pd.DataFrame:
        return self.exclude_input_document(self.documents_data)

    def get_citation_method_data(self, citation_method_data: pd.DataFrame) -> pd.DataFrame:
        return citation_method_data.loc[self.document_id].pipe(self.exclude_input_document)

    def get_co_citation_analysis_data(self) -> pd.DataFrame:
        return self.get_citation_method_data(self.co_citation_analysis_data)

    def get_bibliographic_coupling_data(self) -> pd.DataFrame:
        return self.get_citation_method_data(self.bibliographic_coupling_data)

    def get_info_matrix(self) -> pd.DataFrame:
        return (
            self.filter_documents_data()
            .loc[:, self.info_cols]
            .assign(
                co_citation_analysis=self.get_co_citation_analysis_data(),
                bibliographic_coupling=self.get_bibliographic_coupling_data(),
            )
        )

    def get_feature_matrix(self) -> pd.DataFrame:
        return (
            self.filter_documents_data()
            .loc[:, self.feature_cols]
            .assign(
                co_citation_analysis_rank=self.get_co_citation_analysis_data().rank(
                    ascending=False
                ),
                bibliographic_coupling_rank=self.get_bibliographic_coupling_data().rank(
                    ascending=False
                ),
            )
        )

    def shares_arxiv_label(
        self,
        candidate_document_labels: list[str],
    ) -> bool:
        return any(label in candidate_document_labels for label in self.input_document.labels)

    @staticmethod
    def boolean_to_int(boolean: bool) -> int:
        return int(boolean)

    def get_labels(self) -> pd.Series:
        return (
            self.filter_documents_data()
            .loc[:, "arxiv_labels"]
            .apply(self.shares_arxiv_label)
            .apply(self.boolean_to_int)
            .rename("label")
        )

    def get_model_data(self) -> CitationModelData:
        return CitationModelData(
            self.input_document,
            self.get_info_matrix(),
            self.get_feature_matrix(),
            self.get_labels(),
        )
