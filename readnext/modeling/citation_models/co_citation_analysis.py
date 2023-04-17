import pandas as pd

from readnext.data.config import DataPaths
from readnext.modeling.citation_models.base import (
    compute_n_most_common,
    count_common_values_pairwise,
    fill_values_df,
    lookup_n_most_common,
)


def count_common_citations_pairwise(
    df: pd.DataFrame, document_id_1: int, document_id_2: int
) -> int:
    return count_common_values_pairwise(df, "citations", document_id_1, document_id_2)


def fill_citations_df(
    df: pd.DataFrame, first_row: int | None = None, last_row: int | None = None
) -> pd.DataFrame:
    return fill_values_df(df, count_common_citations_pairwise, first_row, last_row)


def compute_n_most_common_citations(
    df: pd.DataFrame,
    input_document_id: int,
    n: int | None = None,
) -> pd.Series:
    return compute_n_most_common(
        df,
        input_document_id,
        "citations",
        count_common_citations_pairwise,
        "num_common_citations",
        n,
    )


def lookup_n_most_common_citations(
    df: pd.DataFrame,
    input_document_id: int,
    n: int | None = None,
) -> pd.Series:
    return lookup_n_most_common(df, input_document_id, "num_common_citations", n)


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )

    # option 1: pre-compute all pairwise counts, computation at training time
    fill_citations_df(documents_authors_labels_citations_most_cited)

    # top_n_citations = lookup_n_most_common_citations(citations_df, citations_df.index[10])

    # option 2: compute pairwise counts for a single document on the fly, computation at
    # inference time
    # top_n_citations = compute_n_most_common_citations(
    #     documents_authors_labels_citations, document_id_1
    # )

    # pd.merge(
    #     top_n_citations,
    #     documents_authors_labels_citations[["document_id", "title", "arxiv_labels"]],
    #     left_index=True,
    #     right_on="document_id",
    # )


if __name__ == "__main__":
    main()
