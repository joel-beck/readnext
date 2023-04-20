"""
Add arxiv categories as labels to all documents chunks. Keep only documents that have at
least one label in the Computer Science domain.
"""

from pathlib import Path

import pandas as pd

from readnext.config import DataPaths


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add arxiv tags/categories as labels as a new column. Split space separated labels
    into a list.
    """
    return df.assign(arxiv_labels=lambda df: df["categories"].apply(lambda x: x.split()))


def keep_cs_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only labels of the computer science domain."""
    return df.assign(
        arxiv_labels=lambda df: df["arxiv_labels"].apply(
            lambda x: [label for label in x if label.startswith("cs.")]
        )
    )


def remove_non_cs_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove documents without any labels in the Computer Science domain, i.e. with at
    least one list element in the `arxiv_labels` column.
    """
    return df.loc[lambda df: df["arxiv_labels"].apply(len) > 0]


def merge_labels_chunk(filepath: Path, chunk_index: int) -> None:
    documents_preprocessed_chunk: pd.DataFrame = pd.read_pickle(filepath)
    arxiv_id_labels = pd.read_pickle(DataPaths.arxiv.id_labels_pkl)

    documents_labels_chunk = (
        documents_preprocessed_chunk.merge(arxiv_id_labels, on="arxiv_id", how="left")
        .pipe(add_labels)
        .pipe(keep_cs_labels)
        .pipe(remove_non_cs_documents)
    )

    documents_labels_chunk.to_pickle(
        f"{DataPaths.merged.documents_labels_chunk_stem}_{chunk_index}.pkl"
    )


def main() -> None:
    filename_pattern = "documents_preprocessed_chunks_*.pkl"

    for chunk_index, filepath in enumerate(
        DataPaths.merged.documents_labels_chunk_stem.parent.glob(filename_pattern), 1
    ):
        print(f"Merging labels for file {filepath.name}")
        merge_labels_chunk(filepath, chunk_index)


if __name__ == "__main__":
    main()
