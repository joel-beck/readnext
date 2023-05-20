import pandas as pd


def add_citation_feature_rank_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rank columns for publication date, document citation count, and author citation
    count to the dataframe.
    """
    return df.assign(
        publication_date_rank=df["publication_date"].rank(ascending=False),
        citationcount_document_rank=df["citationcount_document"].rank(ascending=False),
        citationcount_author_rank=df["citationcount_author"].rank(ascending=False),
    )


def set_missing_publication_dates_to_max_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the publication date rank to the maxiumum rank (number of documents in
    dataframe) for documents with a missing publication date.
    """
    return df.assign(publication_date_rank=df["publication_date_rank"].fillna(len(df)))
