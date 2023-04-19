from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from readnext.evaluation.metrics import average_precision


@dataclass
class Document:
    document_id: int
    title: str
    author: str
    labels: list[str]


@dataclass
class CitationModelData:
    input_document: Document
    feature_matrix: pd.DataFrame
    labels: pd.Series


@dataclass
class CitationModelDataFromId:
    document_id: int
    documents_data: pd.DataFrame
    co_citation_analysis_data: pd.DataFrame
    bibliographic_coupling_data: pd.DataFrame
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "publication_date_rank",
            "citationcount_document_rank",
            "citationcount_author_rank",
        ]
    )
    document: Document = field(init=False)

    def __post_init__(self) -> None:
        title = str(self.documents_data.loc[self.document_id, "title"])
        author = str(self.documents_data.loc[self.document_id, "author"])
        labels = cast(list[str], self.documents_data.loc[self.document_id, "arxiv_labels"])

        self.document = Document(self.document_id, title, author, labels)

    def exclude_input_document(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df.index != self.document_id]

    def filter_documents_data(self) -> pd.DataFrame:
        return self.exclude_input_document(self.documents_data)

    def filter_citation_method_data(self, citation_method_data: pd.DataFrame) -> pd.DataFrame:
        return citation_method_data.loc[self.document_id].pipe(self.exclude_input_document)

    def filter_co_citation_analysis_data(self) -> pd.DataFrame:
        return self.filter_citation_method_data(self.co_citation_analysis_data)

    def filter_bibliographic_coupling_data(self) -> pd.DataFrame:
        return self.filter_citation_method_data(self.bibliographic_coupling_data)

    def get_feature_matrix(self) -> pd.DataFrame:
        return (
            self.filter_documents_data()
            .loc[:, self.feature_cols]
            .assign(
                co_citation_analysis_ranks=self.filter_co_citation_analysis_data().rank(
                    ascending=False
                ),
                bibliographic_coupling_ranks=self.filter_bibliographic_coupling_data().rank(
                    ascending=False
                ),
            )
        )

    def shares_arxiv_label(
        self,
        candidate_document_labels: list[str],
    ) -> bool:
        return any(label in candidate_document_labels for label in self.document.labels)

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
            self.document,
            self.get_feature_matrix(),
            self.get_labels(),
        )


def add_feature_rank_cols(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.assign(
        publication_date_rank=df["publication_date"].rank(ascending=False),
        citationcount_document_rank=df["citationcount_document"].rank(ascending=False),
        citationcount_author_rank=df["citationcount_author"].rank(ascending=False),
    )


def set_missing_publication_dates_to_max_rank(df: pd.DataFrame) -> pd.DataFrame:
    # set publication_date_rank to maxiumum rank (number of documents in dataframe) for
    # documents with missing publication date
    return df.assign(publication_date_rank=df["publication_date_rank"].fillna(len(df)))


def select_top_n(
    citation_model_data: CitationModelData, feature_name: str, n: int = 20
) -> pd.Series:
    return citation_model_data.feature_matrix[feature_name].sort_values(ascending=True).head(n)


def add_labels(top_n: pd.Series, labels: pd.Series) -> pd.DataFrame:
    return pd.merge(top_n, labels, left_index=True, right_index=True)


def eval_top_n(
    citation_model_data: CitationModelData,
    feature_name: str,
    metric: Callable[[Sequence[int]], float] = average_precision,
    n: int = 20,
) -> float:
    top_n = select_top_n(citation_model_data, feature_name, n)
    top_n_with_labels = add_labels(top_n, citation_model_data.labels)
    return metric(top_n_with_labels["label"])  # type: ignore
