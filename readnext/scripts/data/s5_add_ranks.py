"""
Add rank features for global document characteristics (publication date, document
citation count and author citation count) to the dataset and select a subset of the most
cited documents for the final dataset.

Note that the rank features must be added after subsetting the dataframe to prevent gaps
within the rankings!
"""

import polars as pl

from readnext.config import DataPaths, MagicNumbers
from readnext.utils.aliases import DocumentsFrame
from readnext.utils.io import write_df_to_parquet


def select_most_cited_documents(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Select a subset of the most cited documents from the full documents data set.
    """
    return df.sort(["citationcount_document"], descending=True).head(
        MagicNumbers.documents_frame_final_size
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

    documents_frame: DocumentsFrame = (
        pl.scan_parquet(DataPaths.merged.documents_authors_labels_citations)
        .pipe(select_most_cited_documents)
        .pipe(add_citation_feature_rank_columns)
        .select(output_columns)
        .collect()
    )

    write_df_to_parquet(documents_frame, DataPaths.merged.documents_frame)


if __name__ == "__main__":
    main()
