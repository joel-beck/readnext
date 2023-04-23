import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.modeling.citation_models import (
    compute_n_most_common,
    compute_values_df,
    count_common_values_pairwise,
    lookup_n_most_common,
)


def count_common_references_pairwise(
    df: pd.DataFrame, document_id_1: int, document_id_2: int
) -> int:
    return count_common_values_pairwise(df, "references", document_id_1, document_id_2)


def fill_references_df(
    df: pd.DataFrame, first_row: int | None = None, last_row: int | None = None
) -> pd.DataFrame:
    return compute_values_df(df, count_common_references_pairwise, first_row, last_row)


def compute_n_most_common_references(
    df: pd.DataFrame,
    input_document_id: int,
    n: int | None = None,
) -> pd.Series:
    return compute_n_most_common(
        df,
        input_document_id,
        "references",
        count_common_references_pairwise,
        "num_common_references",
        n,
    )


def lookup_n_most_common_references(
    df: pd.DataFrame,
    input_document_id: int,
    n: int | None = None,
) -> pd.Series:
    return lookup_n_most_common(df, input_document_id, "num_common_references", n)


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )

    # takes roughly 2 minutes
    bibliographic_coupling_most_cited = fill_references_df(
        documents_authors_labels_citations_most_cited
    )

    bibliographic_coupling_most_cited.to_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_most_cited_pkl
    )

    # 8 MB of memory usage
    # references_df.memory_usage(deep=True).sum() / 1e6


if __name__ == "__main__":
    main()
