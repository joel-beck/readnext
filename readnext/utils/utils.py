from collections.abc import Hashable
from typing import TypeVar

from readnext.modeling import DocumentScore

TKey = TypeVar("TKey", bound=Hashable)
TValue = TypeVar("TValue")


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


def sort_document_scores(document_scores: list[DocumentScore]) -> list[DocumentScore]:
    return sorted(document_scores, key=lambda x: x.score, reverse=True)
