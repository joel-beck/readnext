"""
Preprocess all document chunks. Add ranks for the document's citation count and
influential citation count. Keep only documents where the arxiv id is provided - to
merge arxiv tags as labels by the arxiv id later on.
"""

from pathlib import Path

import polars as pl

from readnext.config import DataPaths
from readnext.utils import write_df_to_parquet, get_arxiv_url_from_arxiv_id


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


def main() -> None:
    output_columns = [
        "d3_document_id",
        "d3_author_id",
        "title",
        "author",
        "publication_date",
        "citationcount_document",
        "abstract",
        "semanticscholar_id",
        "semanticscholar_url",
        "semanticscholar_tags",
        "arxiv_id",
        "arxiv_url",
        "arxiv_labels",
    ]

    documents = pl.scan_parquet(DataPaths.d3.documents_parquet, n_rows=1000).collect()
    arxiv_labels = pl.scan_parquet(DataPaths.d3.arxiv_labels_parquet, n_rows=1000).collect()

    documents = documents.rename(
        {
            "corpusid": "d3_document_id",
            "publicationdate": "publication_date",
            "year": "publication_year",
            "citationcount": "citationcount_document",
            "url": "semanticscholar_url",
            "s2fieldsofstudy": "semanticscholar_tags",
        }
    )

    arxiv_labels = (
        arxiv_labels.rename({"id": "arxiv_id", "categories": "arxiv_labels"})
        .select(["arxiv_id", "arxiv_labels"])
        .with_columns(arxiv_url=pl.col("arxiv_id").apply(get_arxiv_url_from_arxiv_id))
    )

    # TODO:
    # 1. Extract author id, author name and arxiv id from dictionary
    # 2. Convert to long format with one row per author
    # 3. Merge Arxiv labels based on Arxiv id
    # 4. Filter to keep only documents with Arxiv id within computer science
    # 5. Write merged dataframe to parquet


if __name__ == "__main__":
    main()
