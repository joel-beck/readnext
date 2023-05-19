from typing import Protocol

from spacy.tokens.doc import Doc


class SpacyModel(Protocol):
    """Protocol for the Spacy Language model."""

    def __call__(self, document: str) -> Doc:
        ...
