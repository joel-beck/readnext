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


def add_citation_feature_rank_columns(df: pl.LazyFrame) -> pl.LazyFrame:
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


def select_most_cited_documents(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Select a subset of the most cited documents from the full documents data set.
    """
    return df.sort(["citationcount_document"], descending=True).head(
        DataPaths.merged.most_cited_subset_size
    )


def main() -> None:
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
        pl.scan_parquet(DataPaths.merged.documents_authors_labels_citations)
        .pipe(add_citation_feature_rank_columns)
        .pipe(select_most_cited_documents)
        .select(output_columns)
        .collect()
    )

    write_df_to_parquet(documents_data, DataPaths.merged.documents_data)


if __name__ == "__main__":
    main()
