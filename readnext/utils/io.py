from pathlib import Path

import pandas as pd


def save_df_to_pickle(df: pd.DataFrame, path: Path) -> None:
    df.to_pickle(path)


def load_df_from_pickle(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path)
