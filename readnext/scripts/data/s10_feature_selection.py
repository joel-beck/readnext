"""
Select only the relevant subset of columns from the full documents data set. Add rank
features for global document characteristics (publication date, document citation count
and author citation count).
"""

import polars as pl

from readnext.config import DataPaths
from readnext.utils import (
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
    read_df_from_parquet,
    write_df_to_parquet,
)


def year_to_first_day_of_year(year: int) -> str:
    """
    Convert a year to the first day of that year.
    """
    return f"{year}-01-01"


def fill_missing_publication_dates_with_year(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fill missing publication dates with the publication year if it exists. The first
    January is chosen as the publication date within this year.
    """
    return df.with_columns(
        publication_date=pl.when(pl.col("publication_date").is_null())
        .then(
            pl.when(pl.col("publication_year").is_not_null())
            .then(pl.col("publication_year").apply(year_to_first_day_of_year))
            .otherwise(pl.col("publication_date"))
        )
        .otherwise(pl.col("publication_date"))
    )


def add_citation_feature_rank_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add rank columns for publication date, document citation count, and author citation
    count to the dataframe.
    """
    return df.with_columns(
        publication_date_rank=pl.col("publication_date").rank(descending=True, method="average"),
        citationcount_document_rank=pl.col("citationcount_document").rank(
            descending=True, method="average"
        ),
        citationcount_author_rank=pl.col("citationcount_author").rank(
            descending=True, method="average"
        ),
    )


def add_identifier_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add Semanticscholar ID and Arxiv URL as additional identifiers for each document.
    """
    return df.with_columns(
        semanticscholar_id=pl.col("semanticscholar_url").apply(
            get_semanticscholar_id_from_semanticscholar_url
        ),
        arxiv_url=pl.col("arxiv_id").apply(get_arxiv_url_from_arxiv_id),
    )


def rename_d3_identifiers(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add `d3` prefix to document and author id within the D3 dataset.
    """
    return df.rename({"document_id": "d3_document_id", "author_id": "d3_author_id"})


def main() -> None:
    documents_authors_labels_citations_most_cited: pl.DataFrame = read_df_from_parquet(
        DataPaths.merged.documents_authors_labels_citations_most_cited_parquet
    )

    output_columns = [
        "d3_document_id",
        "d3_author_id",
        "title",
        "author",
        "publication_date",
        "publication_date_rank",
        "citationcount_document",
        "citationcount_document_rank",
        "citationcount_author",
        "citationcount_author_rank",
        "citations",
        "references",
        "abstract",
        "semanticscholar_id",
        "semanticscholar_url",
        "semanticscholar_tags",
        "arxiv_id",
        "arxiv_url",
        "arxiv_labels",
    ]

    documents_data = (
        documents_authors_labels_citations_most_cited.pipe(fill_missing_publication_dates_with_year)
        .pipe(add_citation_feature_rank_columns)
        .pipe(add_identifier_columns)
        .pipe(rename_d3_identifiers)
        .select(output_columns)
    )

    write_df_to_parquet(documents_data, DataPaths.merged.documents_data_parquet)


if __name__ == "__main__":
    main()
