"""
Add arxiv categories as labels to all documents chunks. Keep only documents that have at
least one label in the Computer Science domain.
"""

from pathlib import Path

import polars as pl

from readnext.config import DataPaths
from readnext.utils import read_df_from_parquet, setup_progress_bar, write_df_to_parquet


def add_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add arxiv tags/categories as labels as a new column. Split space separated labels
    into a list.
    """
    return df.with_columns(arxiv_labels=pl.col("categories").apply(lambda x: x.split()))


def keep_cs_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only labels of the computer science domain."""
    return df.with_columns(
        arxiv_labels=pl.col("arxiv_labels").apply(
            lambda x: [label for label in x if label.startswith("cs.")]
        )
    )


def remove_non_cs_documents(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove documents without any labels in the Computer Science domain, i.e. with at
    least one list element in the `arxiv_labels` column.
    """
    return df.filter(pl.col("arxiv_labels").apply(len) > 0)


def merge_labels_chunk(filepath: Path, chunk_index: int) -> None:
    documents_preprocessed_chunk: pl.DataFrame = read_df_from_parquet(filepath)
    arxiv_id_labels = read_df_from_parquet(DataPaths.arxiv.id_labels_parquet)

    documents_labels_chunk = (
        documents_preprocessed_chunk.join(arxiv_id_labels, on="arxiv_id", how="left")
        .pipe(add_labels)
        .pipe(keep_cs_labels)
        .pipe(remove_non_cs_documents)
    )

    write_df_to_parquet(
        documents_labels_chunk,
        Path(f"{DataPaths.merged.documents_labels_chunk_stem}_{chunk_index}.pkl"),
    )


def main() -> None:
    dirpath_documents_labels_chunks = DataPaths.merged.documents_labels_chunk_stem.parent
    filename_pattern = "documents_preprocessed_chunks_*.pkl"
    matching_files = sorted(dirpath_documents_labels_chunks.glob(filename_pattern))

    with setup_progress_bar() as progress_bar:
        for chunk_index, filepath in progress_bar.track(
            enumerate(matching_files, 1), total=len(matching_files)
        ):
            print(f"Merging labels for file {filepath.name}")
            merge_labels_chunk(filepath, chunk_index)


if __name__ == "__main__":
    main()
