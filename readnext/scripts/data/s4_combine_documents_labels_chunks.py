"""Combine all data chunks with document and label information into a single dataframe."""

import pandas as pd

from readnext.config import DataPaths
from readnext.utils import load_df_from_pickle, setup_progress_bar, write_df_to_pickle


def main() -> None:
    path_documents_labels = DataPaths.merged.documents_labels_pkl
    filename_pattern = "documents_labels_chunks_*.pkl"
    matching_files = sorted(path_documents_labels.parent.glob(filename_pattern))

    df_list = []

    with setup_progress_bar() as progress_bar:
        for filepath in progress_bar.track(matching_files, total=len(matching_files)):
            print(f"Reading File {filepath.name}")
            df_chunk = load_df_from_pickle(filepath)
            df_list.append(df_chunk)

    print("\nConcatenating Dataframes...")
    df_combined = pd.concat(df_list)

    print(f"Writing to {path_documents_labels}")
    write_df_to_pickle(df_combined, path_documents_labels)


if __name__ == "__main__":
    main()
