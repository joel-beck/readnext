"""Combine all data chunks with document and label information into a single dataframe."""

import polars as pl

from readnext.config import DataPaths
from readnext.utils import read_df_from_parquet, setup_progress_bar, write_df_to_parquet


def main() -> None:
    path_documents_labels = DataPaths.merged.documents_labels
    filename_pattern = "documents_labels_chunks_*.pkl"
    matching_files = sorted(path_documents_labels.parent.glob(filename_pattern))

    df_list = []

    with setup_progress_bar() as progress_bar:
        for filepath in progress_bar.track(matching_files, total=len(matching_files)):
            print(f"Reading File {filepath.name}")
            df_chunk = read_df_from_parquet(filepath)
            df_list.append(df_chunk)

    print("\nConcatenating Dataframes...")
    df_combined = pl.concat(df_list)

    print(f"Writing to {path_documents_labels}")
    write_df_to_parquet(df_combined, path_documents_labels)


if __name__ == "__main__":
    main()
