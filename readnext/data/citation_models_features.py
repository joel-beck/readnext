import polars as pl


def add_citation_feature_rank_cols(df: pl.DataFrame) -> pl.DataFrame:
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


def set_missing_publication_dates_to_max_rank(df: pl.DataFrame) -> pl.DataFrame:
    """
    Set the publication date rank to the maxiumum rank (number of documents in
    dataframe) for documents with a missing publication date.
    """
    return df.with_columns(
        publication_date_rank=pl.when(pl.col("publication_date_rank").is_null())
        .then(len(df))
        .otherwise(pl.col("publication_date_rank"))
    )
