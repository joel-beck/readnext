"""
Filter a subset of the most cited documents from the full documents data and store it in
a separate file.
"""

import pandas as pd

from readnext.config import DataPaths
from readnext.utils import load_df_from_pickle, save_df_to_pickle


def main() -> None:
    documents_authors_labels_references: pd.DataFrame = load_df_from_pickle(
        DataPaths.merged.documents_authors_labels_citations_pkl
    )

    documents_authors_labels_citations_most_cited = documents_authors_labels_references.sort_values(
        by="citationcount_document", ascending=False
    ).iloc[: DataPaths.merged.most_cited_subset_size]

    save_df_to_pickle(
        documents_authors_labels_citations_most_cited,
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl,
    )


if __name__ == "__main__":
    main()
