"""
Combine all chunks from parallel requests for citation information into a single
dataframe.
"""

import polars as pl

from readnext.config import DataPaths
from readnext.utils import read_df_from_parquet, setup_progress_bar, write_df_to_parquet


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
            df_chunk = read_df_from_parquet(filepath)
            df_list.append(df_chunk)

    print("\nConcatenating Dataframes...")
    df_combined = pl.concat(df_list)

    # find all rows with empty lists in the citations or references columns
    # could be potential data quality issues

    # df_combined.loc[
    #     (df_combined["citations"].apply(len) == 0) | (df_combined["references"].apply(len) == 0)
    # ]

    write_df_to_parquet(df_combined, path_documents_authors_labels_citations)


if __name__ == "__main__":
    main()
