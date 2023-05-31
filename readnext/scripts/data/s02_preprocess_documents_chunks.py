"""
Preprocess all document chunks. Add ranks for the document's citation count and
influential citation count. Keep only documents where the arxiv id is provided - to
merge arxiv tags as labels by the arxiv id later on.
"""

from pathlib import Path

import polars as pl

from readnext.config import DataPaths
from readnext.utils import add_rank, read_df_from_parquet, setup_progress_bar, write_df_to_parquet


def flatten_list_of_dicts(list_of_dicts: list[dict], key: str) -> list[str]:
    return [d[key] for d in list_of_dicts] if list_of_dicts is not None else []


def remove_duplicates(list_: list[str]) -> list[str]:
    return list(set(list_))


def extract_arxiv_id(df: pl.DataFrame) -> pl.Series:
    # keep only dictionaries where arxiv id is not None
    return df["externalids"].apply(lambda d: d["ArXiv"])


def filter_by_tag(df: pl.DataFrame, tag: str) -> pl.DataFrame:
    return df.filter(pl.col("tags").apply(lambda x: tag in x))


def preprocess_document_chunk(filepath: Path, chunk_index: int) -> None:
    documents_chunk: pl.DataFrame = read_df_from_parquet(filepath)

    documents_ranked = documents_chunk.with_columns(
        citationcount_rank=add_rank(documents_chunk["citationcount"]),
        influentialcitationcount_rank=add_rank(documents_chunk["influentialcitationcount"]),
    )

    documents_long_format = (
        documents_ranked.with_columns(
            author_id=pl.col("authors").apply(lambda x: flatten_list_of_dicts(x, "authorId")),
            author_name=pl.col("authors").apply(lambda x: flatten_list_of_dicts(x, "name")),
            tags=pl.col("s2fieldsofstudy")
            .apply(lambda x: flatten_list_of_dicts(x, "category"))
            .apply(remove_duplicates),
        )
        .explode(["author_id", "author_name"])
        .drop_nulls(subset=["author_id", "author_name"])
        .with_columns(
            author_id=pl.col("author_id").cast(pl.Int64),
        )
        .pipe(filter_by_tag, tag="Computer Science")
    )

    documents_long_format_arxiv = documents_long_format.with_columns(
        arxiv_id=extract_arxiv_id(documents_long_format)
    ).filter(pl.col("arxiv_id").is_not_null())

    write_df_to_parquet(
        documents_long_format_arxiv,
        Path(f"{DataPaths.d3.documents.preprocessed_chunks_stem}_{chunk_index}.pkl"),
    )


def main() -> None:
    dirpath_documents_preprocessed_chunks = DataPaths.d3.documents.preprocessed_chunks_stem.parent
    filename_pattern = "*_documents_chunks_*.pkl"
    matching_files = sorted(dirpath_documents_preprocessed_chunks.glob(filename_pattern))

    with setup_progress_bar() as progress_bar:
        for chunk_index, filepath in progress_bar.track(
            enumerate(matching_files, 1), total=len(matching_files)
        ):
            print(f"Preprocessing file {filepath.name}")
            preprocess_document_chunk(filepath, chunk_index)


if __name__ == "__main__":
    main()
