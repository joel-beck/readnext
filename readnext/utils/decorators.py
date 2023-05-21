import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, Concatenate, Literal, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def loading_message(path: Path, data_type: Any) -> str:
    return f"Loading {data_type} from {path.name}..."


def writing_message(path: Path, data_type: Any) -> str:
    return f"Writing {data_type} to {path.name}..."


def loader_decorator_factory(
    data_type: Literal["Object", "Data Frame"]
) -> Callable[[Callable[Concatenate[Path, P], R]], Callable[Concatenate[Path, P], R]]:
    """
    Create a decorator for loading data operations.

    This decorator factory generates a decorator intended for functions that load data,
    with the Path to the data file as their first argument. The decorated function
    can have any number of additional positional and keyword arguments.

    The decorator prints a loading message before calling the decorated function
    and a checkmark after the function returns.

    Args:
        data_type: The type of the data to be loaded, must be either "Object" or "Data Frame".

    Returns:
        A decorator that can be used to decorate data loading functions.

    Usage:
        @loader_decorator_factory(data_type="Object")
        def load_object_from_pickle(path: Path) -> Any:
            ...
    """

    def decorator(func: Callable[Concatenate[Path, P], R]) -> Callable[Concatenate[Path, P], R]:
        @functools.wraps(func)
        def wrapper(path: Path, *args: P.args, **kwargs: P.kwargs) -> R:
            print(loading_message(path=path, data_type=data_type), end=" ")

            result = func(path, *args, **kwargs)

            print("✅")
            return result

        return wrapper

    return decorator


def writer_decorator_factory(
    data_type: Literal["Object", "Data Frame"]
) -> Callable[[Callable[Concatenate[Any, Path, P], R]], Callable[Concatenate[Any, Path, P], R]]:
    """
    Create a decorator for writing data operations.

    This decorator factory generates a decorator intended for functions that write data,
    with the data to be written as their first argument, and the Path to the data file
    as their second argument. The decorated function can have any number of additional
    positional and keyword arguments.

    The decorator prints a writing message before calling the decorated function and a
    checkmark after the function returns.

    Args:
        data_type: The type of the data to be written, must be either "Object" or "Data
        Frame".

    Returns:
        A decorator that can be used to decorate data writing functions.

    Usage:
        @writer_decorator_factory(data_type="Object") def write_object_to_pickle(obj:
        Any, path: Path) -> None:
            ...
    """

    def decorator(
        func: Callable[Concatenate[Any, Path, P], R]
    ) -> Callable[Concatenate[Any, Path, P], R]:
        @functools.wraps(func)
        def wrapper(obj: Any, path: Path, *args: P.args, **kwargs: P.kwargs) -> R:
            print(writing_message(path=path, data_type=data_type), end=" ")

            result = func(obj, path, *args, **kwargs)

            print("✅")
            return result

        return wrapper

    return decorator


object_loader = loader_decorator_factory(data_type="Object")
dataframe_loader = loader_decorator_factory(data_type="Data Frame")
object_writer = writer_decorator_factory(data_type="Object")
dataframe_writer = writer_decorator_factory(data_type="Data Frame")
