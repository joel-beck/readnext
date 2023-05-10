from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from readnext.utils import setup_progress_bar


def test_setup_progress_bar() -> None:
    progress_bar = setup_progress_bar()

    assert isinstance(progress_bar, Progress)
    assert len(progress_bar.columns) == 7
    assert isinstance(progress_bar.columns[0], TextColumn)
    assert isinstance(progress_bar.columns[1], BarColumn)
    assert isinstance(progress_bar.columns[2], MofNCompleteColumn)
    assert isinstance(progress_bar.columns[3], TextColumn)
    assert isinstance(progress_bar.columns[4], TimeElapsedColumn)
    assert isinstance(progress_bar.columns[5], TextColumn)
    assert isinstance(progress_bar.columns[6], TimeRemainingColumn)
