"""
Preprocess the D3 authors dataset and merge it with the D3 documents dataset via the D3
author id.

Add the author citationcount to the dataset and select the most popular author for each
document.
"""

import polars as pl

from readnext.config import DataPaths
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def rename_authors_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Rename columns of the authors dataset to the desired output column names.
    """
    return df.rename({"authorid": "d3_author_id", "citationcount": "citationcount_author"})


def select_authors_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Select only the columns that are needed in the output dataframe.
    """
    return df.select(["d3_author_id", "citationcount_author"]).with_columns(
        pl.col("d3_author_id").cast(pl.Int64)
    )


def merge_authors(documents_labels: pl.DataFrame, authors: pl.DataFrame) -> pl.DataFrame:
    """
    Merge the authors dataset with the documents dataset via the D3 author id.
    """
    return documents_labels.join(authors, how="left", on="d3_author_id")


def select_most_popular_author(df: pl.DataFrame) -> pl.DataFrame:
    """
    Sort by author popularity metric within each document from highest to lowest and
    keep first row. Then sort the unique documents again by document citation count.
    """
    return (
        df.sort(["d3_document_id", "citationcount_author"], descending=True)
        .unique(subset=["d3_document_id"], keep="first")
        .sort("citationcount_document", descending=True)
    )


def main() -> None:
    output_columns = [
        "d3_document_id",
        "d3_author_id",
        "title",
        "author",
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "abstract",
        "semanticscholar_id",
        "semanticscholar_url",
        "semanticscholar_tags",
        "arxiv_id",
        "arxiv_url",
        "arxiv_labels",
    ]

    authors = (
        pl.scan_parquet(DataPaths.raw.authors_parquet)
        .pipe(rename_authors_columns)
        .pipe(select_authors_features)
        .collect()
    )

    documents_authors_labels = (
        read_df_from_parquet(DataPaths.merged.documents_labels)
        .pipe(merge_authors, authors)
        .pipe(select_most_popular_author)
        .select(output_columns)
        .drop_nulls()
    )

    write_df_to_parquet(documents_authors_labels, DataPaths.merged.documents_authors_labels)


if __name__ == "__main__":
    main()
