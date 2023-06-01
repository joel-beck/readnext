import pickle
from pathlib import Path
from typing import Any

import polars as pl

from readnext.utils.decorators import (
    dataframe_reader,
    dataframe_writer,
    object_reader,
    object_writer,
)


@object_reader
def read_object_from_pickle(path: Path) -> Any:
    """Read any Python object from a pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore


@object_writer
def write_object_to_pickle(
    obj: Any,
    path: Path,
) -> None:
    """Write any Python object to a pickle file."""
    with path.open("wb") as f:
        pickle.dump(obj, f)


@dataframe_reader
def read_df_from_parquet(path: Path) -> pl.DataFrame:
    """Read a Polars DataFrame from a parquet file."""
    return pl.read_parquet(path)


@dataframe_writer
def write_df_to_parquet(
    df: pl.DataFrame,
    path: Path,
) -> None:
    """Write a Polars DataFrame to a parquet file."""
    df.write_parquet(path)
