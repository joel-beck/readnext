"""
Add author information to documents and labels. Match authors with documents by the
author id.
"""

import polars as pl

from readnext.config import DataPaths
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def select_most_popular_author(
    df: pl.DataFrame, author_popularity_metric: str = "citationcount_author"
) -> pl.DataFrame:
    """
    Sort by author popularity metric within each document from highest to lowest and
    keep first row. Then sort the unique documents again by document citation count.
    """
    return (
        df.sort(["document_id", author_popularity_metric], descending=True)
        .unique(subset=["document_id"], keep="first")
        .sort("citationcount_document", descending=True)
    )


def remove_incomplete_data(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop_nulls(
        subset=["publication_year", "citationcount_document", "citationcount_author", "abstract"]
    )


def main() -> None:
    documents_labels: pl.DataFrame = read_df_from_parquet(DataPaths.merged.documents_labels_pkl)
    authors: pl.DataFrame = read_df_from_parquet(DataPaths.d3.authors.full_parquet)

    output_columns = [
        "d3_document_id",
        "d3_author_id",
        "title",
        "author",
        "author_aliases",
        "publication_date",
        "publication_year",
        "citationcount_document",
        "citationcount_document_rank",
        "influentialcitationcount_document",
        "influentialcitationcount_document_rank",
        "referencecount_document",
        "citationcount_author",
        "citationcount_author_rank",
        "hindex_author",
        "hindex_author_rank",
        "papercount_author",
        "papercount_author_rank",
        "abstract",
        "arxiv_id",
        "arxiv_labels",
        "semanticscholar_url",
        "semanticscholar_tags",
    ]

    documents_authors_labels_long = (
        documents_labels.join(
            authors_ranked,
            how="left",
            left_on="author_id",
            right_on="authorid",
            # TODO: How to assign different suffixes to left and right dataframe?
            suffix=("_document", "_author"),
        )
        .rename(
            {
                "corpusid": "document_id",
                "author_name": "author",
                "aliases": "author_aliases",
                "publicationdate": "publication_date",
                "year": "publication_year",
                "citationcount_rank_document": "citationcount_document_rank",
                "influentialcitationcount": "influentialcitationcount_document",
                "influentialcitationcount_rank": "influentialcitationcount_document_rank",
                "referencecount": "referencecount_document",
                "citationcount_rank_author": "citationcount_author_rank",
                "hindex": "hindex_author",
                "hindex_rank": "hindex_author_rank",
                "papercount": "papercount_author",
                "papercount_rank": "papercount_author_rank",
                "url": "semanticscholar_url",
                "tags": "semanticscholar_tags",
            }
        )
        .select(output_columns)
    )

    # select one row per document with most cited author and consider only complete document
    # observations in Computer Science
    documents_authors_labels = documents_authors_labels_long.pipe(
        select_most_popular_author, author_popularity_metric="citationcount_author"
    ).pipe(remove_incomplete_data)

    write_df_to_parquet(documents_authors_labels, DataPaths.merged.documents_authors_labels_pkl)


if __name__ == "__main__":
    main()
