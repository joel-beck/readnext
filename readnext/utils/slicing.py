from collections.abc import Callable, Hashable
from typing import Concatenate, ParamSpec, TypeVar

import polars as pl

from readnext.utils.progress_bar import rich_progress_bar

P = ParamSpec("P")
TKey = TypeVar("TKey", bound=Hashable)
TValue = TypeVar("TValue")


def concatenate_sliced_dataframes(
    df: pl.DataFrame,
    slice_function: Callable[Concatenate[pl.DataFrame, P], pl.DataFrame],
    slice_size: int,
    progress_bar_description: str = "",
    *function_args: P.args,
    **function_kwargs: P.kwargs,
) -> pl.DataFrame:
    total_size = df.height
    num_slices = total_size // slice_size

    with rich_progress_bar() as progress_bar:
        return pl.concat(
            [
                slice_function(
                    df.slice(next_index, slice_size),
                    *function_args,
                    **function_kwargs,
                )
                for next_index in progress_bar.track(
                    range(0, total_size, slice_size),
                    total=num_slices,
                    description=progress_bar_description,
                )
            ]
        )


# maybe todo: figure out correct typing for DataFrame and LazyFrame to reduce duplicated
# code with previous function
def concatenate_sliced_lazyframes(
    df: pl.LazyFrame,
    slice_function: Callable[Concatenate[pl.LazyFrame, P], pl.DataFrame],
    slice_size: int,
    progress_bar_description: str = "",
    *function_args: P.args,
    **function_kwargs: P.kwargs,
) -> pl.DataFrame:
    total_size = df.collect().height
    num_slices = total_size // slice_size

    with rich_progress_bar() as progress_bar:
        return pl.concat(
            [
                slice_function(
                    df.slice(next_index, slice_size),
                    *function_args,
                    **function_kwargs,
                )
                for next_index in progress_bar.track(
                    range(0, total_size, slice_size),
                    total=num_slices,
                    description=progress_bar_description,
                )
            ]
        )


def slice_mapping(
    mapping: dict[TKey, TValue],
    size: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> dict[TKey, TValue]:
    """
    Subset a dictionary by numeric indices. If size is provided, it takes precedence
    over start and end.
    """

    if size is not None:
        return {key: value for i, (key, value) in enumerate(mapping.items()) if i < size}

    if start is None:
        start = 0

    if end is None:
        end = len(mapping)

    return {key: value for i, (key, value) in enumerate(mapping.items()) if start <= i < end}
