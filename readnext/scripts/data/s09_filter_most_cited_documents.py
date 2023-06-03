"""
Filter a subset of the most cited documents from the full documents data and store it in
a separate file.
"""

import pandas as pd
import polars as pl

from readnext.config import DataPaths
from readnext.utils import write_df_to_parquet


def main() -> None:
    documents_authors_labels_references: pd.DataFrame = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations
    )

    documents_authors_labels_citations_most_cited = documents_authors_labels_references.sort_values(
        by="citationcount_document", ascending=False
    ).iloc[: DataPaths.merged.most_cited_subset_size]

    write_df_to_parquet(
        pl.from_pandas(documents_authors_labels_citations_most_cited),
        DataPaths.merged.documents_authors_labels_citations_most_cited_parquet,
    )


if __name__ == "__main__":
    main()
