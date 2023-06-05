"""
Preprocess the D3 documents dataset and the Arxiv dataset.

Merge the arxiv metadata with the D3 dataset via the arxiv id. Add arxiv labels as new
feature to the dataset which are later used as ground-truth labels for the recommender
system.

Reduces the dataset size significantly since only a subset of documents in the D3
dataset contain an arxiv id as external identifier.
"""

from collections.abc import Sequence
from typing import TypedDict

import polars as pl

from readnext.config import DataPaths, MagicNumbers
from readnext.utils import (
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
    write_df_to_parquet,
)


class SemanticScholarTag(TypedDict):
    category: str
    source: str


def rename_arxiv_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Rename arxiv columns to the desired output dataframe column names.
    """
    return df.rename({"id": "arxiv_id", "categories": "arxiv_labels"})


def filter_cs_labels(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filter out all documents that do not have an Arxiv label of the Computer Science
    domain.
    """
    return df.filter(pl.col("arxiv_labels").str.contains(r"^cs\."))


def add_arxiv_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add arxiv urls and labels to the dataframe. The arxiv labels are split from
    space-separated strings into a list of strings.
    """
    return df.with_columns(
        arxiv_url=pl.col("arxiv_id").apply(get_arxiv_url_from_arxiv_id),
        arxiv_labels=pl.col("arxiv_labels").str.split(" "),
    )


def select_arxiv_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Select only the columns that are needed in the output dataframe.
    """
    return df.select(["arxiv_id", "arxiv_url", "arxiv_labels"])


def rename_document_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Rename document columns to the desired output dataframe column names.
    """
    return df.rename(
        {
            "corpusid": "d3_document_id",
            "publicationdate": "publication_date",
            "year": "publication_year",
            "citationcount": "citationcount_document",
            "url": "semanticscholar_url",
            "s2fieldsofstudy": "semanticscholar_tags",
        }
    )


def select_most_cited_documents(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Select the top milltion most cited documents from the full documents data set.
    """
    return df.sort(["citationcount_document"], descending=True).head(
        MagicNumbers.documents_data_intermediate_cutoff
    )


def extract_unique_semanticscholar_tags(
    semanticscholar_tags: Sequence[SemanticScholarTag],
) -> list[str]:
    return list({tag["category"] for tag in semanticscholar_tags})


def format_semanticscholar_tags(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Format the semanticscholar tags column from a struct into a list of strings.
    """
    return df.with_columns(
        semanticscholar_tags=pl.col("semanticscholar_tags").apply(
            extract_unique_semanticscholar_tags
        )
    )


def year_to_first_day_of_year(year: int) -> str:
    """
    Convert a year to the first day of that year.
    """
    return f"{year}-01-01"


def fill_missing_publication_dates_with_year(df: pl.LazyFrame) -> pl.LazyFrame:
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


def convert_author_columns_to_long_format(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Split dictionaries of author names and author ids into separate rows with one
    row per author.
    """
    return df.explode("authors").with_columns(
        d3_author_id=pl.col("authors").struct.field("authorId").cast(pl.Int64),
        author=pl.col("authors").struct.field("name"),
    )


def add_semanticscholar_ids(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add semanticscholar ids to the dataframe.
    """
    return df.with_columns(
        semanticscholar_id=pl.col("semanticscholar_url").apply(
            get_semanticscholar_id_from_semanticscholar_url
        )
    )


def add_non_missing_arxiv_ids(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add arxiv ids to the dataframe. Remove documents where the arxiv id is missing in
    the D3 dataset.
    """
    return df.with_columns(arxiv_id=pl.col("externalids").struct.field("ArXiv")).drop_nulls(
        "arxiv_id"
    )


def merge_arxiv_data(documents_data: pl.DataFrame, arxiv_data: pl.DataFrame) -> pl.DataFrame:
    """
    Merge arxiv labels with the document dataframe. Note that merging operations require
    DataFrames instead of LazyFrames!
    """
    return documents_data.join(arxiv_data, on="arxiv_id", how="left")


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

    arxiv_data = (
        pl.scan_parquet(DataPaths.raw.arxiv_labels_parquet)
        .pipe(rename_arxiv_columns)
        .pipe(filter_cs_labels)
        .pipe(add_arxiv_features)
        .pipe(select_arxiv_features)
        .collect()
    )

    documents_data = (
        pl.scan_parquet(DataPaths.raw.documents_parquet)
        .pipe(rename_document_columns)
        .pipe(select_most_cited_documents)
        .pipe(format_semanticscholar_tags)
        .pipe(fill_missing_publication_dates_with_year)
        .pipe(convert_author_columns_to_long_format)
        .pipe(add_semanticscholar_ids)
        .pipe(add_non_missing_arxiv_ids)
        .collect()
    )

    merged_data = (
        documents_data.pipe(merge_arxiv_data, arxiv_data).select(output_columns).drop_nulls()
    )

    write_df_to_parquet(merged_data, DataPaths.merged.documents_labels)


if __name__ == "__main__":
    main()
