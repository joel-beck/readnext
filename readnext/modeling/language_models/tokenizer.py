from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar, cast

import torch
from spacy.language import Language
from spacy.tokens.doc import Doc
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.modeling.document_info import DocumentsInfo
from readnext.utils import (
    load_object_from_pickle,
    save_object_to_pickle,
    setup_progress_bar,
)

# each document is represented as a list of tokens
Tokens: TypeAlias = list[str]
TokensMapping: TypeAlias = dict[int, Tokens]

# each document is represented as a tensor of token ids
TokenIds: TypeAlias = list[int]
TokensIdMapping: TypeAlias = dict[int, list[int]]
TokensTensorMapping: TypeAlias = dict[int, torch.Tensor]

TTorchTokenizer = TypeVar("TTorchTokenizer", bound=BertTokenizerFast | LongformerTokenizerFast)


@dataclass
class ListTokenizer(ABC):
    """Base class to tokenize abstracts into a list of string tokens."""

    documents_info: DocumentsInfo

    @abstractmethod
    def tokenize(self) -> TokensMapping:
        ...

    @staticmethod
    def save_tokens_mapping(path: Path, tokens_mapping: TokensMapping) -> None:
        """Save a mapping of document ids to tokens to a pickle file."""
        save_object_to_pickle(tokens_mapping, path)

    @staticmethod
    def load_tokens_mapping(path: Path) -> TokensMapping:
        """Load a mapping of document ids to tokens from a pickle file."""
        return load_object_from_pickle(path)  # type: ignore


@dataclass
class TextProcessingSteps:
    """
    Defines all possible text processing steps that can be applied to a document.
    """

    remove_non_alphanumeric: bool = True
    remove_non_ascii: bool = True
    remove_digits: bool = True
    remove_punctuation: bool = True
    remove_whitespace: bool = True
    remove_stopwords: bool = True
    to_lowercase: bool = True
    lemmatize: bool = True


@dataclass
class SpacyTokenizer(ListTokenizer):
    """Tokenize abstracts using spacy into a list of string tokens."""

    documents_info: DocumentsInfo
    spacy_model: Language
    text_processing_steps: TextProcessingSteps = TextProcessingSteps()  # noqa

    def to_spacy_doc(self, document: str) -> Doc:
        """Converts a single abstract into a spacy document."""
        return self.spacy_model(document)

    def clean_spacy_doc(
        self,
        spacy_doc: Doc,
    ) -> Tokens:
        """
        Cleans a single spacy document by removing stopwords, punctuation, and
        non-alphanumeric tokens.
        """
        clean_tokens = []
        # use for loop instead of list comprehension to allow disabling of individual filters
        for token in spacy_doc:
            # initialize token_text such that lemmatisation and lowercasing can be
            # applied independently of each other
            token_text = token.text

            if self.text_processing_steps.remove_non_alphanumeric and not token.is_alpha:
                continue

            if self.text_processing_steps.remove_non_ascii and not token.is_ascii:
                continue

            if self.text_processing_steps.remove_digits and token.is_digit:
                continue

            if self.text_processing_steps.remove_punctuation and token.is_punct:
                continue

            if self.text_processing_steps.remove_whitespace and token.is_space:
                continue

            if self.text_processing_steps.remove_stopwords and token.is_stop:
                continue

            if self.text_processing_steps.lemmatize:
                token_text = token.lemma_

            if self.text_processing_steps.to_lowercase:
                token_text = token_text.lower()

            clean_tokens.append(token_text)

        return clean_tokens

    def clean_document(self, document: str) -> Tokens:
        """Tokenizes and cleans a single abstract."""
        spacy_doc = self.to_spacy_doc(document)
        return self.clean_spacy_doc(spacy_doc)

    def tokenize(self) -> TokensMapping:
        """
        Tokenizes and cleans multiple abstracts. Gnereates a mapping of document ids to
        tokens.
        """
        tokenized_abstracts_mapping = {}

        with setup_progress_bar() as progress_bar:
            for document_id, abstract in progress_bar.track(
                zip(self.documents_info.document_ids, self.documents_info.abstracts),
                total=len(self.documents_info.document_ids),
                description=f"{self.__class__.__name__}:",
            ):
                tokenized_abstracts_mapping[document_id] = self.clean_document(abstract)

        return tokenized_abstracts_mapping


@dataclass
class TensorTokenizer(ABC, Generic[TTorchTokenizer]):
    """Base class to tokenize abstracts into a tensor of token ids."""

    documents_info: DocumentsInfo
    tensor_tokenizer: TTorchTokenizer

    @abstractmethod
    def tokenize(self) -> TokensIdMapping:
        ...

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

    def tokenize(self) -> TokensIdMapping:
        """
        Tokenizes multiple abstracts into token ids. Generates a mapping of document ids
        to token ids.
        """
        token_ids = self.tensor_tokenizer(
            self.documents_info.abstracts,
            # BERT takes 512 dimensional tensors as input
            max_length=512,
            # truncate longer documents down to 512 tokens
            truncation=True,
            # pad shorter documents up to 512 tokens
            padding=True,
            # return_tensors="pt",
        )["input_ids"]

        return dict(zip(self.documents_info.document_ids, token_ids))  # type: ignore


@dataclass
class LongformerTokenizer(TensorTokenizer):
    """Tokenize abstracts using Longformer into a tensor of token ids."""

    tensor_tokenizer: LongformerTokenizerFast

    def tokenize(self) -> TokensIdMapping:
        """
        Tokenizes multiple abstracts into token ids. Generates a mapping of document ids
        to token ids.
        """
        token_ids = self.tensor_tokenizer(
            self.documents_info.abstracts,
            # Longformer can handle up to 4096 tokens
            max_length=4096,
            truncation=True,
            padding=True,
        )["input_ids"]

        return dict(zip(self.documents_info.document_ids, token_ids))  # type: ignore
