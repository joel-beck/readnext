import pandas as pd


def add_feature_rank_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        publication_date_rank=df["publication_date"].rank(ascending=False),
        citationcount_document_rank=df["citationcount_document"].rank(ascending=False),
        citationcount_author_rank=df["citationcount_author"].rank(ascending=False),
    )


def set_missing_publication_dates_to_max_rank(df: pd.DataFrame) -> pd.DataFrame:
    # set publication_date_rank to maxiumum rank (number of documents in dataframe) for
    # documents with missing publication date
    return df.assign(publication_date_rank=df["publication_date_rank"].fillna(len(df)))
