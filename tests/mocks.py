import numpy as np
import torch
from numpy.typing import NDArray

from readnext.utils import (
    BertModelProtocol,
    FastTextModelProtocol,
    LongformerModelProtocol,
    Tokens,
    TorchModelOutputProtocol,
    Word2VecModelProtocol,
    WordVectorsProtocol,
)


def word2vec_model_mock() -> Word2VecModelProtocol:
    class Word2VecModelMock:
        def __getitem__(self, document_tokens: Tokens) -> NDArray:  # noqa: ARG002
            return np.array([1, 2, 3])

        def __contains__(self, document_tokens: Tokens) -> bool:  # noqa: ARG002
            return True

    return Word2VecModelMock()


def word_vectors_model_mock() -> WordVectorsProtocol:
    class WordVectorsModelMock:
        def __getitem__(self, document_tokens: Tokens) -> NDArray:  # noqa: ARG002
            return np.array([1, 2, 3])

        def __contains__(self, document_tokens: Tokens) -> bool:  # noqa: ARG002
            return True

    return WordVectorsModelMock()


def fasttext_model_mock() -> FastTextModelProtocol:
    class FastTextModelModelMock:
        @property
        def wv(self) -> WordVectorsProtocol:
            return word_vectors_model_mock()

    return FastTextModelModelMock()


def torch_model_output_mock() -> TorchModelOutputProtocol:
    class TorchModelOutputMock:
        @property
        def last_hidden_state(self) -> torch.Tensor:
            return torch.Tensor(torch.Tensor([1, 2, 3]))

    return TorchModelOutputMock()


def bert_model_mock() -> BertModelProtocol:
    class BertModelMock:
        def __call__(
            self, token_ids_tensor: torch.Tensor  # noqa: ARG002
        ) -> TorchModelOutputProtocol:
            return torch_model_output_mock()

    return BertModelMock()


def longformer_model_mock() -> LongformerModelProtocol:
    class LongformerModelMock:
        def __call__(
            self, token_ids_tensor: torch.Tensor  # noqa: ARG002
        ) -> TorchModelOutputProtocol:
            return torch_model_output_mock()

    return LongformerModelMock()
