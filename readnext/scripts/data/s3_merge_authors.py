"""
Add author information to documents and labels. Match authors with documents by the
author id.
"""

import polars as pl

from readnext.config import DataPaths
from readnext.utils import read_df_from_parquet, write_df_to_parquet


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

    documents_labels = read_df_from_parquet(DataPaths.merged.documents_labels)
    authors = (
        pl.scan_parquet(DataPaths.raw.authors_parquet)
        .rename({"authorid": "d3_author_id", "citationcount": "citationcount_author"})
        .select(["d3_author_id", "citationcount_author"])
        .with_columns(pl.col("d3_author_id").cast(pl.Int64))
        .collect()
    )

    documents_authors_labels = (
        documents_labels.join(authors, how="left", on="d3_author_id")
        .select(output_columns)
        .pipe(select_most_popular_author)
        .drop_nulls()
    )

    write_df_to_parquet(documents_authors_labels, DataPaths.merged.documents_authors_labels)


if __name__ == "__main__":
    main()
