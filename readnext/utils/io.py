from pathlib import Path

import polars as pl


# @dataframe_reader
def read_df_from_parquet(path: Path) -> pl.DataFrame:
    """Read a Polars DataFrame from a parquet file."""
    return pl.read_parquet(path)


# @dataframe_writer
def write_df_to_parquet(
    df: pl.DataFrame,
    path: Path,
) -> None:
    """Write a Polars DataFrame to a parquet file."""
    df.write_parquet(path)
