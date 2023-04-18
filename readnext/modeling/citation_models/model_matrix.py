from readnext.modeling.config import ResultsPaths
from readnext.data.config import DataPaths
import pandas as pd
from dataclasses import dataclass
from typing import TypeVar

TData = TypeVar("TData", pd.DataFrame, pd.Series)


@dataclass
class Document:
    document_id: int
    title: str
    author: str
    labels: list[str]


@dataclass
class ModelData:
    input_document: Document
    feature_matrix: pd.DataFrame
    labels: pd.Series


def add_publication_date_rank(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.assign(publication_date_rank=df["publication_date"].rank(ascending=False))


def shares_arxiv_label(
    candidate_document_labels: list[str],
    input_document_labels: list[str],
) -> bool:
    return any(label in candidate_document_labels for label in input_document_labels)


def boolean_to_int(boolean: bool) -> int:
    return int(boolean)


def exclude_input_document(
    data: TData,
    input_document_id: int,
) -> TData:
    return data.loc[data.index != input_document_id]


def set_missing_publication_dates_to_max_rank(df: pd.DataFrame) -> pd.DataFrame:
    # set publication_date_rank to maxiumum rank (number of documents in dataframe) for
    # documents with missing publication date
    return df.assign(publication_date_rank=df["publication_date_rank"].fillna(len(df)))


def get_citation_method_data(
    citation_method_data: pd.DataFrame, input_document_id: int
) -> pd.DataFrame:
    return citation_method_data.loc[input_document_id].pipe(
        exclude_input_document, input_document_id=input_document_id
    )


documents_authors_labels_citations_most_cited: pd.DataFrame = (
    pd.read_pickle(DataPaths.merged.documents_authors_labels_citations_most_cited_pkl)
    .set_index("document_id")
    .pipe(add_publication_date_rank)
    .pipe(set_missing_publication_dates_to_max_rank)
)

bibliographic_coupling_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.citation_models.bibliographic_coupling_most_cited_pkl
)

co_citation_analysis_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.citation_models.co_citation_analysis_most_cited_pkl
)

input_document_id = 206594692

input_document_data = documents_authors_labels_citations_most_cited.loc[input_document_id]

input_document = Document(
    document_id=input_document_id,
    title=input_document_data["title"],
    author=input_document_data["author"],
    labels=input_document_data["arxiv_labels"],
)

document_cols = ["title", "author", "arxiv_labels"]
feature_cols = ["publication_date_rank", "citationcount_document_rank", "citationcount_author_rank"]

model_matrix = documents_authors_labels_citations_most_cited.loc[
    :,
    document_cols + feature_cols,
].pipe(exclude_input_document, input_document_id=input_document_id)


labels = (
    model_matrix["arxiv_labels"]
    .apply(shares_arxiv_label, args=(input_document.labels,))
    .apply(boolean_to_int)
)

# documents_authors_labels_citations_most_cited["arxiv_labels"].value_counts().plot.bar()


feature_matrix = model_matrix.loc[:, feature_cols].assign(
    co_citation_analysis_ranks=get_citation_method_data(
        co_citation_analysis_most_cited, input_document_id
    ).rank(ascending=False),
    bibliographic_coupling_ranks=get_citation_method_data(
        bibliographic_coupling_most_cited, input_document_id
    ).rank(ascending=False),
)

model_data = ModelData(input_document=input_document, feature_matrix=feature_matrix, labels=labels)


# TODO: Ranks are not computed correctly!
