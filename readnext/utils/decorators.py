import functools
from typing import Callable, Any, Literal
from pathlib import Path
from enum import Enum


class Operation(Enum):
    LOADING = "Loading"
    WRITING = "Writing"

    def __str__(self) -> str:
        return self.value


def loading_message(path: Path, data_type: Any) -> str:
    return f"Loading {data_type} from {path.name}..."


def writing_message(path: Path, data_type: Any) -> str:
    return f"Writing {data_type} to {path.name}..."


def decorator_factory(
    func: Callable,
    data_type: Literal["Object", "Data Frame"],
    operation: Operation,
) -> Callable:
    @functools.wraps(func)
    def wrapper(path: Path, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        message_func = loading_message if operation == Operation.LOADING else writing_message
        print(message_func(path, data_type), end=" ")

        result = func(path, *args, **kwargs)

        print("✅")
        return result

    return wrapper


object_loader = functools.partial(
    decorator_factory, data_type="Object", operation=Operation.LOADING
)
object_writer = functools.partial(
    decorator_factory, data_type="Object", operation=Operation.WRITING
)
dataframe_loader = functools.partial(
    decorator_factory, data_type="Data Frame", operation=Operation.LOADING
)
dataframe_writer = functools.partial(
    decorator_factory, data_type="Data Frame", operation=Operation.WRITING
)
