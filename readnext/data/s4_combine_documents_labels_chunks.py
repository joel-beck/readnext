"""Combine all data chunks with document and label information into a single dataframe."""

import pandas as pd

from readnext.config import DataPaths


def main() -> None:
    path_documents_labels = DataPaths.merged.documents_labels_pkl
    filename_pattern = "documents_labels_chunks_*.pkl"
    df_list = []

    for filepath in path_documents_labels.parent.glob(filename_pattern):
        print(f"Reading File {filepath.name}")
        df_chunk = pd.read_pickle(filepath)
        df_list.append(df_chunk)

    print("\nConcatenating Dataframes...")
    df_combined = pd.concat(df_list)

    print(f"Writing to {path_documents_labels}")
    df_combined.to_pickle(path_documents_labels)


if __name__ == "__main__":
    main()
