"""
Combine all chunks from parallel requests for citation information into a single
dataframe.
"""

import pandas as pd

from readnext.config import DataPaths
from readnext.utils import setup_progress_bar


def main() -> None:
    path_documents_authors_labels_citations = (
        DataPaths.merged.documents_authors_labels_citations_pkl
    )
    filename_pattern = "documents_authors_labels_citations_chunks_*.pkl"
    matching_files = sorted(path_documents_authors_labels_citations.parent.glob(filename_pattern))

    df_list = []

    with setup_progress_bar() as progress_bar:
        for filepath in progress_bar.track(matching_files, total=len(matching_files)):
            print(f"Reading File {filepath.name}")
            df_chunk = pd.read_pickle(filepath)
            df_list.append(df_chunk)

    print("\nConcatenating Dataframes...")
    df_combined = pd.concat(df_list)

    # find all rows with empty lists in the citations or references columns
    # could be potential data quality issues

    # df_combined.loc[
    #     (df_combined["citations"].apply(len) == 0) | (df_combined["references"].apply(len) == 0)
    # ]

    print(f"Writing to {path_documents_authors_labels_citations}")
    df_combined.to_pickle(path_documents_authors_labels_citations)


if __name__ == "__main__":
    main()
