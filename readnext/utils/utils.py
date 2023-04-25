from collections.abc import Hashable
from typing import TypeVar

TKey = TypeVar("TKey", bound=Hashable)
TValue = TypeVar("TValue")


def slice_mapping(
    mapping: dict[TKey, TValue], start: int | None, end: int | None
) -> dict[TKey, TValue]:
    """Subset a dictionary by numeric indices."""
    if start is None:
        start = 0

    if end is None:
        end = len(mapping)

    return {key: value for i, (key, value) in enumerate(mapping.items()) if start <= i < end}
