from collections.abc import Callable, Sequence

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.metrics import average_precision
from readnext.modeling import LanguageModelData, LanguageModelDataFromId


def select_top_n_ranks(language_model_data: LanguageModelData, n: int = 20) -> pd.DataFrame:
    return language_model_data.embedding_ranks.sort_values("cosine_similarity_rank").head(n)


def move_cosine_similarity_first(df: pd.DataFrame) -> pd.DataFrame:
    return df[["cosine_similarity"] + list(df.columns.drop("cosine_similarity"))]


def add_info_cols(df: pd.DataFrame, info_matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.merge(df, info_matrix, left_index=True, right_index=True)
        .drop("cosine_similarity_rank", axis="columns")
        .pipe(move_cosine_similarity_first)
    )


def add_labels(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    return pd.merge(df, labels, left_index=True, right_index=True)


def display_top_n(
    language_model_data: LanguageModelData,
    n: int = 20,
) -> pd.DataFrame:
    return select_top_n_ranks(language_model_data, n).pipe(
        add_info_cols, language_model_data.info_matrix
    )


def score_top_n(
    language_model_data: LanguageModelData,
    metric: Callable[[Sequence[int]], float] = average_precision,
    n: int = 20,
) -> float:
    top_n_ranks_with_labels = select_top_n_ranks(language_model_data, n).pipe(
        add_labels, language_model_data.integer_labels
    )

    return metric(top_n_ranks_with_labels["label"])  # type: ignore


# evaluation for a single input document
input_document_id = 206594692

documents_authors_labels_citations_most_cited: pd.DataFrame = pd.read_pickle(
    DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
).set_index("document_id")

tfidf_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
)

tfidf_cosine_similarities_most_cited.loc[input_document_id]

language_model_data_from_id = LanguageModelDataFromId(
    document_id=input_document_id,
    documents_data=documents_authors_labels_citations_most_cited,
    cosine_similarity_matrix=tfidf_cosine_similarities_most_cited,
)
language_model_data = language_model_data_from_id.get_model_data()

select_top_n_ranks(language_model_data, n=10)
score_top_n(language_model_data, n=10)
display_top_n(language_model_data, n=10)
