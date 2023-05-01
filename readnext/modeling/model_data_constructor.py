from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from readnext.modeling.document_info import DocumentInfo, DocumentScore


@dataclass(kw_only=True)
class ModelDataConstructor(ABC):
    query_document_id: int
    documents_data: pd.DataFrame
    info_cols: list[str]
    query_document: DocumentInfo = field(init=False)

    def __post_init__(self) -> None:
        self.query_document = self.collect_query_document()

    @abstractmethod
    def extend_info_matrix(self, info_matrix: pd.DataFrame) -> pd.DataFrame:
        ...

    def collect_query_document(self) -> DocumentInfo:
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
        return df.loc[df.index != self.query_document_id]

    def filter_documents_data(self) -> pd.DataFrame:
        return self.exclude_query_document(self.documents_data)

    def get_info_matrix(self) -> pd.DataFrame:
        return self.filter_documents_data().loc[:, self.info_cols]

    def shares_arxiv_label(
        self,
        candidate_document_labels: list[str],
    ) -> bool:
        return any(label in candidate_document_labels for label in self.query_document.arxiv_labels)

    @staticmethod
    def boolean_to_int(boolean: bool) -> int:
        return int(boolean)

    def get_integer_labels(self) -> pd.Series:
        return (
            self.filter_documents_data()
            .loc[:, "arxiv_labels"]
            .apply(self.shares_arxiv_label)
            .apply(self.boolean_to_int)
            .rename("integer_labels")
        )

    def document_scores_to_frame(self, document_scores: list[DocumentScore]) -> pd.DataFrame:
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
        document_scores: list[DocumentScore] = citation_method_data.loc[
            self.query_document_id
        ].item()

        return self.document_scores_to_frame(document_scores)

    def get_co_citation_analysis_scores(self) -> pd.DataFrame:
        return self.get_citation_method_scores(self.co_citation_analysis_scores).rename(
            columns={"score": "co_citation_analysis"}
        )

    def get_bibliographic_coupling_scores(self) -> pd.DataFrame:
        return self.get_citation_method_scores(self.bibliographic_coupling_scores).rename(
            columns={"score": "bibliographic_coupling"}
        )

    def extend_info_matrix(self, info_matrix: pd.DataFrame) -> pd.DataFrame:
        return info_matrix.assign(
            co_citation_analysis=self.get_co_citation_analysis_scores(),
            bibliographic_coupling=self.get_bibliographic_coupling_scores(),
        )

    def get_feature_matrix(self) -> pd.DataFrame:
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
    cosine_similarities: pd.DataFrame
    info_cols: list[str] = field(default_factory=lambda: ["title", "author", "arxiv_labels"])

    def get_cosine_similarity_scores(self) -> pd.DataFrame:
        document_scores: list[DocumentScore] = self.cosine_similarities.loc[
            self.query_document_id
        ].item()

        return self.document_scores_to_frame(document_scores).rename(
            columns={"score": "cosine_similarity"}
        )

    def extend_info_matrix(self, info_matrix: pd.DataFrame) -> pd.DataFrame:
        return info_matrix.assign(
            cosine_similarity=self.get_cosine_similarity_scores(),
        )

    def get_cosine_similarity_ranks(self) -> pd.DataFrame:
        return (
            self.get_cosine_similarity_scores()
            .rank(ascending=False)
            .rename({"cosine_similarity": "cosine_similarity_rank"}, axis="columns")
        )
