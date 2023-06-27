import functools
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Concatenate, Literal, ParamSpec, TypeVar

import polars as pl

TParams = ParamSpec("TParams")
TReturn = TypeVar("TReturn")


def status_update(
    message: str = "Processing...", padding_with: int = 40
) -> Callable[[Callable[TParams, TReturn]], Callable[TParams, TReturn]]:
    """
    Decorator factory for functions that print a status message and the execution time
    before calling the decorated function.

    This decorator factory generates a decorator intended for functions that perform a
    long-running operation. It prints the status message before calling the decorated
    function and a checkmark after the function returns.

    Args:
        message: The status message to be printed.

    Returns:
        A decorator that can be used to decorate long-running functions.

    Usage:
        @status_update(message="Processing...") def long_running_function(*args: P.args,
        **kwargs: P.kwargs) -> R:
            ...
    """

    def decorator(func: Callable[TParams, TReturn]) -> Callable[TParams, TReturn]:
        @functools.wraps(func)
        def wrapper(*args: TParams.args, **kwargs: TParams.kwargs) -> TReturn:
            message_padded = message.ljust(padding_with, ".")
            print(message_padded, end=" ")

            start = perf_counter()

            result = func(*args, **kwargs)

            stop = perf_counter()

            print(f"✅ ({stop - start:.2f} seconds)")
            return result

        return wrapper

    return decorator


def reading_dataframe_message(path: Path) -> str:
    return f"Reading Data Frame from {path.name}..."


def writing_dataframe_message(path: Path) -> str:
    return f"Writing Data Frame to {path.name}..."


def dataframe_reader(func: Callable[[Path], pl.DataFrame]) -> Callable[[Path], pl.DataFrame]:
    """
    Decorator for reading DataFrame operations.

    This decorator is intended for functions that read a dataframe from a file, with the
    Filepath as their argument.

    The decorator prints a writing message before calling the decorated function and a
    checkmark after the function returns.

    Returns:
        The decorated function.

    Usage:
        @dataframe_reader
        def read_df_from_pickle(path: Path) -> pl.Dataframe:
            ...
    """

    @functools.wraps(func)
    def wrapper(path: Path) -> pl.DataFrame:
        print(reading_dataframe_message(path), end=" ")

        df = func(path)

        print("✅")
        return df

    return wrapper


def dataframe_writer(
    func: Callable[[pl.DataFrame, Path], None]
) -> Callable[[pl.DataFrame, Path], None]:
    """
    Decorator for writing DataFrame operations.

    This decorator is intended for functions that write data,
    with the DataFrame to be written as their first argument,
    and the Path to the data file as their second argument.

    The decorator prints a writing message before calling the decorated function
    and a checkmark after the function returns.

    Returns:
        The decorated function.

    Usage:
        @dataframe_writer
        def write_df_to_pickle(df: pl.DataFrame, path: Path) -> None:
            ...
    """

    @functools.wraps(func)
    def wrapper(df: pl.DataFrame, path: Path) -> None:
        print(writing_dataframe_message(path), end=" ")

        func(df, path)

        print("✅")

    return wrapper


def reading_message(path: Path, data_type: str) -> str:
    return f"Reading {data_type} from {path.name}..."


# currently unused, but keep as template for correct typing of decorator factory
def reader_decorator_factory(
    data_type: Literal["Object", "Data Frame"]
) -> Callable[
    [Callable[Concatenate[Path, TParams], TReturn]], Callable[Concatenate[Path, TParams], TReturn]
]:
    def decorator(
        func: Callable[Concatenate[Path, TParams], TReturn]
    ) -> Callable[Concatenate[Path, TParams], TReturn]:
        @functools.wraps(func)
        def wrapper(path: Path, *args: TParams.args, **kwargs: TParams.kwargs) -> TReturn:
            print(reading_message(path=path, data_type=data_type), end=" ")

            result = func(path, *args, **kwargs)

            print("✅")
            return result

        return wrapper

    return decorator


# dataframe_reader = reader_decorator_factory(data_type="Data Frame")
# object_reader = reader_decorator_factory(data_type="Object")
