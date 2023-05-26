"""
Add author information to documents and labels. Match authors with documents by the
author id.
"""

import pandas as pd

from readnext.config import DataPaths
from readnext.utils import add_rank, load_df_from_pickle, write_df_to_pickle


def add_author_ranks(authors: pd.DataFrame) -> pd.DataFrame:
    return authors.assign(
        citationcount_rank=add_rank(authors["citationcount"]),
        hindex_rank=add_rank(authors["hindex"]),
        papercount_rank=add_rank(authors["papercount"]),
    )


def select_most_popular_author(
    df: pd.DataFrame, author_popularity_metric: str = "citationcount_author"
) -> pd.DataFrame:
    """
    Sort by author popularity metric within each document from highest to lowest and
    keep first row. Then sort the unique documents again by document citation count.
    """
    return (
        df.sort_values(["document_id", author_popularity_metric], ascending=False)
        .drop_duplicates(subset=["document_id"], keep="first")
        .sort_values("citationcount_document", ascending=False)
    )


def remove_incomplete_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(
        subset=["publication_year", "citationcount_document", "citationcount_author", "abstract"]
    )


def main() -> None:
    documents_labels: pd.DataFrame = load_df_from_pickle(DataPaths.merged.documents_labels_pkl)
    authors: pd.DataFrame = load_df_from_pickle(DataPaths.d3.authors.full_pkl)

    output_columns = [
        "document_id",
        "author_id",
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

    authors_ranked = add_author_ranks(authors)

    documents_authors_labels_long = (
        documents_labels.merge(
            authors_ranked,
            left_on="author_id",
            right_on="authorid",
            suffixes=("_document", "_author"),
        )
        .rename(
            columns={
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
        .loc[:, output_columns]
    )

    # select one row per document with most cited author and consider only complete document
    # observations in Computer Science
    documents_authors_labels = documents_authors_labels_long.pipe(
        select_most_popular_author, author_popularity_metric="citationcount_author"
    ).pipe(remove_incomplete_data)

    write_df_to_pickle(documents_authors_labels, DataPaths.merged.documents_authors_labels_pkl)


if __name__ == "__main__":
    main()
