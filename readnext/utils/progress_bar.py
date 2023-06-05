from collections.abc import Callable
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm


def setup_progress_bar() -> Progress:
    """Setup a pretty `rich` progress bar."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def tqdm_progress_bar_wrapper(progress_bar: tqdm, func: Callable) -> Callable:
    def foo(*args: Any, **kwargs: Any) -> Any:
        progress_bar.update(1)
        return func(*args, **kwargs)

    return foo
