import pickle
from pathlib import Path
from typing import Any

import polars as pl

from readnext.modeling import DocumentScore
from readnext.utils.decorators import (
    dataframe_reader,
    dataframe_writer,
    object_reader,
    object_writer,
)


@object_writer
def write_object_to_pickle(
    obj: Any,
    path: Path,
) -> None:
    """Write any Python object to a pickle file."""
    with path.open("wb") as f:
        pickle.dump(obj, f)


@object_reader
def read_object_from_pickle(path: Path) -> Any:
    """Read any Python object from a pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)  # type: ignore


def write_df_to_parquet(
    df: pl.DataFrame,
    path: Path,
) -> None:
    """Write a Polars DataFrame to a parquet file."""
    df.write_parquet(path)


@dataframe_reader
def read_df_from_parquet(path: Path) -> pl.DataFrame:
    """Read a Polars DataFrame from a parquet file."""
    return pl.read_parquet(path)


@dataframe_writer
def write_scores_frame_to_parquet(
    df: pl.DataFrame,
    path: Path,
) -> None:
    """
    Write a Dataframe of type `ScoresFrame` to a parquet file.

    A `ScoresFrame` contains two columns: `document_id` and `scores`, where each row in
    `scores` is a list of `DocumentScores`. To write this dataframe to a parquet file,
    we need to serialize the `scores` column and deserialize it when reading the parquet
    file.
    """

    scores_colname = df.columns[1]
    df = df.with_columns(
        pl.col(scores_colname).apply(lambda scores: [score.serialize() for score in scores]),
    )
    df.write_parquet(path)


@dataframe_reader
def read_scores_frame_from_parquet(path: Path) -> pl.DataFrame:
    """
    Read a Dataframe of type `ScoresFrame` from a parquet file by deserializing the
    `scores` column into proper `DocumentScore` objects.

    Note that complex python objects can only be inserted into Polars DataFrames during
    construction and not once the dataframe structure is defined!
    """
    df = pl.read_parquet(path)
    scores_colname = df.columns[1]

    return pl.DataFrame(
        {
            "document_id": df["document_id"],
            scores_colname: [
                [DocumentScore.deserialize(score) for score in scores]
                for scores in df[scores_colname]
            ],
        }
    )
