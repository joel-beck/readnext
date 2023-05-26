from typing import Protocol

import torch
from numpy.typing import NDArray

from readnext.utils.aliases import Tokens


class Word2VecModelProtocol(Protocol):
    """Protocol for the Gensim Word2Vec model / KeyedVectors interface."""

    def __getitem__(self, document_tokens: Tokens) -> NDArray:
        ...

    def __contains__(self, document_tokens: Tokens) -> bool:
        ...


class WordVectorsProtocol(Protocol):
    """Protocol for Gensim Word."""

    def __getitem__(self, document_tokens: Tokens) -> NDArray:
        ...

    def __contains__(self, document_tokens: Tokens) -> bool:
        ...


class FastTextModelProtocol(Protocol):
    """Protocol for the Gensim FastText model."""

    @property
    def wv(self) -> WordVectorsProtocol:
        ...


class TorchModelOutputProtocol(Protocol):
    """Protocol for the output of a transformers pytorch model."""

    @property
    def last_hidden_state(self) -> torch.Tensor:
        ...


class BertModelProtocol(Protocol):
    """Protocol for the transformers pytorch Bert model."""

    def __call__(self, token_ids_tensor: torch.Tensor) -> TorchModelOutputProtocol:
        ...


class LongformerModelProtocol(Protocol):
    """Protocol for the transformers pytorch Longformer model."""

    def __call__(self, token_ids_tensor: torch.Tensor) -> TorchModelOutputProtocol:
        ...
