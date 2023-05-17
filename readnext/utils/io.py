import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from readnext.utils.decorators import (
    dataframe_loader,
    dataframe_writer,
    object_loader,
    object_writer,
)


@object_writer
def write_object_to_pickle(obj: Any, path: Path) -> None:
    """Write any Python object to a pickle file."""
    with path.open("wb") as f:
        pickle.dump(obj, f)


@object_loader
def load_object_from_pickle(path: Path) -> Any:
    """Load any Python object from a pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore


@dataframe_writer
def write_df_to_pickle(df: pd.DataFrame, path: Path) -> None:
    """Write a Pandas DataFrame to a pickle file."""
    df.to_pickle(path)


@dataframe_loader
def load_df_from_pickle(path: Path) -> pd.DataFrame:
    """Load a Pandas DataFrame from a pickle file."""
    return pd.read_pickle(path)
