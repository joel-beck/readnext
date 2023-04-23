from collections.abc import Callable, Sequence
from enum import Enum

import pandas as pd

from readnext.evaluation.metrics import average_precision
from readnext.modeling.citation_models import CitationModelData


class ScoringFeature(Enum):
    publication_date = "publication_date_rank"
    citationcount_document = "citationcount_document_rank"
    citationcount_author = "citationcount_author_rank"
    co_citation_analysis = "co_citation_analysis_rank"
    bibliographic_coupling = "bibliographic_coupling_rank"
    weighted = "weighted_rank"

    def __str__(self) -> str:
        return self.value


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


def select_top_n_ranks(
    citation_model_data: CitationModelData, by: ScoringFeature, n: int = 20
) -> pd.DataFrame:
    # `weighted` option for now computes row sums, i.e. each feature is weighted equally
    if by.name == "weighted":
        ranks_unsorted = citation_model_data.feature_matrix.sum(axis=1).rename("weighted_rank")
    else:
        ranks_unsorted = citation_model_data.feature_matrix[by.value]

    return ranks_unsorted.sort_values().head(n).to_frame()


def add_info_cols(df: pd.DataFrame, info_matrix: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(df, info_matrix, left_index=True, right_index=True)


def add_labels(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    return pd.merge(df, labels, left_index=True, right_index=True)


def display_top_n(
    citation_model_data: CitationModelData,
    by: ScoringFeature,
    n: int = 20,
) -> pd.DataFrame:
    return select_top_n_ranks(citation_model_data, by, n).pipe(
        add_info_cols, citation_model_data.info_matrix
    )


def score_top_n(
    citation_model_data: CitationModelData,
    by: ScoringFeature,
    metric: Callable[[Sequence[int]], float] = average_precision,
    n: int = 20,
) -> float:
    top_n_ranks_with_labels = select_top_n_ranks(citation_model_data, by, n).pipe(
        add_labels, citation_model_data.integer_labels
    )

    return metric(top_n_ranks_with_labels["label"])  # type: ignore
