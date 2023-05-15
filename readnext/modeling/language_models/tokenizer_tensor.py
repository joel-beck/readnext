from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar, cast

import torch
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.modeling.document_info import DocumentsInfo
from readnext.modeling.language_models.tokenizer_list import Tokens, TokensMapping
from readnext.utils import (
    load_object_from_pickle,
    save_object_to_pickle,
)

# each document is represented as a tensor of token ids
TokenIds: TypeAlias = list[int]
TokensIdMapping: TypeAlias = dict[int, list[int]]
TokensTensorMapping: TypeAlias = dict[int, torch.Tensor]

TTorchTokenizer = TypeVar("TTorchTokenizer", bound=BertTokenizerFast | LongformerTokenizerFast)


@dataclass
class TensorTokenizer(ABC, Generic[TTorchTokenizer]):
    """Base class to tokenize document abstracts into a tensor of token ids."""

    documents_info: DocumentsInfo
    tensor_tokenizer: TTorchTokenizer

    @abstractmethod
    def tokenize_into_ids(self, document: str) -> TokenIds:
        """Tokenizes one or multiple document abstracts into token ids."""

    @abstractmethod
    def tokenize(self) -> TokensIdMapping:
        """
        Tokenizes multiple document abstracts into token ids. Generates a mapping of
        document ids to token ids.
        """

    def ids_to_tokens(self, token_ids: TokenIds) -> Tokens:
        """Converts a list of token ids into a list of tokens."""
        return cast(list[str], self.tensor_tokenizer.convert_ids_to_tokens(token_ids))

    def id_mapping_to_tokens_mapping(self, tokens_id_mapping: TokensIdMapping) -> TokensMapping:
        """
        Converts a mapping of document ids to token ids into a mapping of document ids
        to tokens.
        """
        return {
            document_id: self.ids_to_tokens(token_ids)
            for document_id, token_ids in tokens_id_mapping.items()
        }

    @staticmethod
    def save_tokens_mapping(path: Path, tokens_id_mapping: TokensIdMapping) -> None:
        """Save a mapping of document ids to token ids to a pickle file."""
        save_object_to_pickle(tokens_id_mapping, path)

    @staticmethod
    def load_tokens_mapping(path: Path) -> TokensIdMapping:
        """Load a mapping of document ids to token ids from a pickle file."""
        return load_object_from_pickle(path)  # type: ignore


@dataclass
class BERTTokenizer(TensorTokenizer):
    """Tokenize abstracts using BERT into a tensor of token ids."""

    tensor_tokenizer: BertTokenizerFast

    def tokenize_into_ids(self, document: str | list[str]) -> TokenIds:
        return self.tensor_tokenizer(
            document,
            # BERT takes 512 dimensional tensors as input
            max_length=512,
            # truncate longer documents down to 512 tokens
            truncation=True,
            # pad shorter documents up to 512 tokens
            padding=True,
        )[
            "input_ids"
        ]  # type: ignore

    def tokenize(self) -> TokensIdMapping:
        token_ids = self.tokenize_into_ids(self.documents_info.abstracts)
        return dict(zip(self.documents_info.document_ids, token_ids))  # type: ignore


@dataclass
class LongformerTokenizer(TensorTokenizer):
    """Tokenize abstracts using Longformer into a tensor of token ids."""

    tensor_tokenizer: LongformerTokenizerFast

    def tokenize_into_ids(self, document: str | list[str]) -> TokenIds:
        return self.tensor_tokenizer(
            document,
            # Longformer can handle up to 4096 tokens
            max_length=4096,
            truncation=True,
            padding=True,
        )[
            "input_ids"
        ]  # type: ignore

    def tokenize(self) -> TokensIdMapping:
        token_ids = self.tokenize_into_ids(self.documents_info.abstracts)

        return dict(zip(self.documents_info.document_ids, token_ids))  # type: ignore
