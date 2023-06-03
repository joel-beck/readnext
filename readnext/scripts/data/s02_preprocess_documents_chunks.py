"""
Preprocess all document chunks. Add ranks for the document's citation count and
influential citation count. Keep only documents where the arxiv id is provided - to
merge arxiv tags as labels by the arxiv id later on.
"""

from pathlib import Path

import polars as pl

from readnext.config import DataPaths
from readnext.utils import (
    write_df_to_parquet,
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
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


def flatten_list_of_dicts(list_of_dicts: list[dict], key: str) -> list[str]:
    return [d[key] for d in list_of_dicts] if list_of_dicts is not None else []


def preprocess_document_chunk(filepath: Path, chunk_index: int) -> None:
    documents_long_format = (
        documents_ranked.with_columns(
            author_id=pl.col("authors").apply(lambda x: flatten_list_of_dicts(x, "authorId")),
            author_name=pl.col("authors").apply(lambda x: flatten_list_of_dicts(x, "name")),
        )
        .explode(["author_id", "author_name"])
        .drop_nulls(subset=["author_id", "author_name"])
        .with_columns(
            author_id=pl.col("author_id").cast(pl.Int64),
        )
    )


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

    documents = pl.scan_parquet(DataPaths.d3.documents_parquet, n_rows=10000)
    arxiv_labels = pl.scan_parquet(DataPaths.d3.arxiv_labels_parquet, n_rows=10000)

    arxiv_labels = (
        arxiv_labels.rename({"id": "arxiv_id", "categories": "arxiv_labels"})
        .filter(pl.col("arxiv_labels").str.contains("cs.", literal=True))
        .with_columns(
            arxiv_url=pl.col("arxiv_id").apply(get_arxiv_url_from_arxiv_id),
            arxiv_labels=pl.col("arxiv_labels").str.split(" "),
        )
        .select(["arxiv_id", "arxiv_url", "arxiv_labels"])
    )

    documents_labels = (
        documents.rename(
            {
                "corpusid": "d3_document_id",
                "publicationdate": "publication_date",
                "year": "publication_year",
                "citationcount": "citationcount_document",
                "url": "semanticscholar_url",
                "s2fieldsofstudy": "semanticscholar_tags",
            }
        )
        .with_columns(
            semanticscholar_id=pl.col("semanticscholar_url").apply(
                get_semanticscholar_id_from_semanticscholar_url
            ),
            arxiv_id=pl.col("externalids").apply(lambda d: d["ArXiv"], skip_nulls=False),
        )
        .drop_nulls(subset=["arxiv_id"])
        .join(arxiv_labels, on="arxiv_id", how="inner")
        .collect()
    )

    (
        documents_labels.pipe(fill_missing_publication_dates_with_year)
        .with_columns(
            d3_author_id=pl.col("authors").apply(lambda x: flatten_list_of_dicts(x, "authorId")),
            author=pl.col("authors").apply(lambda x: flatten_list_of_dicts(x, "name")),
        )
        .explode(["d3_author_id", "author"])
        .with_columns(pl.col("d3_author_id").cast(pl.Int64))
        .select(output_columns)
    )

    # TODO:
    # 2. Simplify semanticscholar tags to list of strings
    # 3. Remove linting errors
    # 4. Make code modular by using functions that return expression
    # 5. Write merged dataframe to parquet


if __name__ == "__main__":
    main()
