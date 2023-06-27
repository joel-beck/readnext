from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import polars as pl
from tqdm import tqdm
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.utils.aliases import DocumentsFrame, TokenIds, TokenIdsFrame, Tokens, TokensFrame
from readnext.utils.progress_bar import tqdm_progress_bar_wrapper

TTorchTokenizer = TypeVar("TTorchTokenizer", bound=BertTokenizerFast | LongformerTokenizerFast)


@dataclass
class TorchTokenizer(ABC, Generic[TTorchTokenizer]):
    """Base class to tokenize document abstracts into a tensor of token ids."""

    torch_tokenizer: TTorchTokenizer

    @abstractmethod
    def tokenize_into_ids(self, document: str) -> TokenIds:
        """Tokenizes one or multiple document abstracts into token ids."""

    def tokenize(self, documents_frame: DocumentsFrame) -> TokenIdsFrame:
        """
        Tokenizes multiple document abstracts into token ids. Generates a polars
        dataframe with two columns named `d3_document_id` and `token_ids`.
        """
        abstracts_frame = documents_frame.select(["d3_document_id", "abstract"])

        with tqdm(total=len(abstracts_frame)) as progress_bar:
            token_ids_frame = abstracts_frame.with_columns(
                token_ids=pl.col("abstract").apply(
                    tqdm_progress_bar_wrapper(progress_bar, self.tokenize_into_ids)
                )
            )

        return token_ids_frame.drop("abstract")

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

    def tokenize_into_ids(self, document: str | list[str]) -> TokenIds:
        return self.torch_tokenizer(
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


@dataclass
class LongformerTokenizer(TorchTokenizer):
    """Tokenize abstracts using Longformer into a tensor of token ids."""

    torch_tokenizer: LongformerTokenizerFast

    def tokenize_into_ids(self, document: str | list[str]) -> TokenIds:
        return self.torch_tokenizer(
            document,
            # Longformer can handle up to 4096 tokens
            max_length=4096,
            truncation=True,
            padding=True,
        )[
            "input_ids"
        ]  # type: ignore
