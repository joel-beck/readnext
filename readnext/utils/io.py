import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_object_to_pickle(obj: Any, path: Path, verbose: bool = True) -> None:  # type: ignore # noqa
    """Save any Python object to a pickle file."""
    if verbose:
        print(f"Saving object to {path.name}...")

    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_object_from_pickle(path: Path, verbose: bool = True) -> Any:  # type: ignore # noqa
    """Load any Python object from a pickle file."""
    if verbose:
        print(f"Loading object from {path.name}...")

    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore


def save_df_to_pickle(df: pd.DataFrame, path: Path, verbose: bool = True) -> None:
    """Save a Pandas DataFrame to a pickle file."""
    if verbose:
        print(f"Saving DataFrame to {path.name}...")

    df.to_pickle(path)


def load_df_from_pickle(path: Path, verbose: bool = True) -> pd.DataFrame:
    """Load a Pandas DataFrame from a pickle file."""
    if verbose:
        print(f"Loading DataFrame from {path.name}...")

    return pd.read_pickle(path)
