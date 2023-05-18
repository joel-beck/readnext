import functools
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class Operation(Enum):
    LOADING = "Loading"
    WRITING = "Writing"

    def __str__(self) -> str:
        return self.value


def loading_message(path: Path, data_type: Any) -> str:
    return f"Loading {data_type} from {path.name}..."


def writing_message(path: Path, data_type: Any) -> str:
    return f"Writing {data_type} to {path.name}..."


# TODO: Type correctly to identify multiple arguments of wrapped function correctly!
def decorator_factory(
    func: Callable,
    data_type: Literal["Object", "Data Frame"],
    operation: Operation,
) -> Callable:
    @functools.wraps(func)
    def wrapper(path: Path, *args: Any, **kwargs: Any) -> Any:
        message_func = loading_message if operation == Operation.LOADING else writing_message
        print(message_func(path=path, data_type=data_type), end=" ")

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
