from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar, cast, overload

import polars as pl
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.utils.aliases import DocumentsFrame, TokenIds, TokenIdsFrame, Tokens, TokensFrame

TTorchTokenizer = TypeVar("TTorchTokenizer", bound=BertTokenizerFast | LongformerTokenizerFast)


@dataclass
class TorchTokenizer(ABC, Generic[TTorchTokenizer]):
    """Base class to tokenize document abstracts into a tensor of token ids."""

    torch_tokenizer: TTorchTokenizer
    max_tokens: int

    @overload
    def tokenize_into_ids(self, documents: str) -> TokenIds:
        ...

    @overload
    def tokenize_into_ids(self, documents: list[str]) -> list[TokenIds]:
        ...

    def tokenize_into_ids(self, documents: str | list[str]) -> TokenIds | list[TokenIds]:
        """
        Tokenizes a single document or a list of documents into token ids using a torch
        tokenizer.
        """
        return self.torch_tokenizer(
            documents,
            max_length=self.max_tokens,
            # pad shorter documents up to 512 tokens
            padding=True,
            # truncate longer documents down to 512 tokens
            truncation=True,
        ).input_ids

    def tokenize(self, documents_frame: DocumentsFrame) -> TokenIdsFrame:
        """
        Tokenizes multiple document abstracts into token ids. Generates a polars
        dataframe with two columns named `d3_document_id` and `token_ids`.
        """
        token_ids = self.tokenize_into_ids(documents_frame["abstract"].to_list())

        return documents_frame.select("d3_document_id").with_columns(token_ids=pl.Series(token_ids))

    def tokens_to_token_ids(self, tokens: Tokens) -> TokenIds:
        """Converts a list of tokens into a list of token ids."""
        return cast(list[int], self.torch_tokenizer.convert_tokens_to_ids(tokens))

    def tokens_frame_to_token_ids_frame(self, tokens_frame: TokensFrame) -> TokenIdsFrame:
        """
        Converts a dataframe with a `tokens` column into a dataframe with a `token_ids`
        column.
        """
        return tokens_frame.with_columns(
            token_ids=pl.col("tokens").apply(self.tokens_to_token_ids)
        ).drop("tokens")

    def token_ids_to_tokens(self, token_ids: TokenIds) -> Tokens:
        """Converts a list of token ids into a list of tokens."""
        return cast(list[str], self.torch_tokenizer.convert_ids_to_tokens(token_ids))

    def token_ids_frame_to_tokens_frame(self, token_ids_frame: TokenIdsFrame) -> TokensFrame:
        """
        Converts a dataframe with a `token_ids` column into a dataframe with a `tokens`
        column.
        """
        return token_ids_frame.with_columns(
            tokens=pl.col("token_ids").apply(self.token_ids_to_tokens)
        ).drop("token_ids")


@dataclass
class BERTTokenizer(TorchTokenizer):
    """Tokenize abstracts using BERT into a tensor of token ids."""

    torch_tokenizer: BertTokenizerFast
    # BERT takes 512 dimensional tensors as input
    max_tokens: int = 512


@dataclass
class LongformerTokenizer(TorchTokenizer):
    """Tokenize abstracts using Longformer into a tensor of token ids."""

    torch_tokenizer: LongformerTokenizerFast
    # Longformer can handle up to 4096 tokens
    max_tokens: int = 4096
