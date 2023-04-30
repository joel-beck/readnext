import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_object_to_pickle(obj: Any, path: Path) -> None:  # type: ignore # noqa: ANN401
    """Save any Python object to a pickle file."""
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_object_from_pickle(path: Path) -> Any:  # type: ignore # noqa: ANN401
    """Load any Python object from a pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore


def save_df_to_pickle(df: pd.DataFrame, path: Path) -> None:
    """Save a Pandas DataFrame to a pickle file."""
    df.to_pickle(path)


def load_df_from_pickle(path: Path) -> pd.DataFrame:
    """Load a Pandas DataFrame from a pickle file."""
    return pd.read_pickle(path)
