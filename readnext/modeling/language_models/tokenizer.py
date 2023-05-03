from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from spacy.language import Language
from spacy.tokens.doc import Doc
from transformers import BertTokenizerFast

# do not import from .language_models to avoid circular imports
from readnext.modeling import DocumentsInfo
from readnext.utils import (
    load_object_from_pickle,
    save_object_to_pickle,
    setup_progress_bar,
)

# each document is represented as a list of tokens
DocumentTokens: TypeAlias = list[str]
DocumentTokensMapping: TypeAlias = dict[int, DocumentTokens]

# each document is represented as a string of tokens separated by whitespace
DocumentString: TypeAlias = str
DocumentStringMapping: TypeAlias = dict[int, DocumentString]

# each document is represented as a tensor of token ids
DocumentsTokensTensor: TypeAlias = torch.Tensor
DocumentsTokensTensorMapping: TypeAlias = dict[int, DocumentsTokensTensor]


@dataclass
class ListTokenizer(ABC):
    """Base class to tokenize abstracts into a list of string tokens."""

    documents_info: DocumentsInfo

    @abstractmethod
    def tokenize(self) -> DocumentTokensMapping:
        ...

    @staticmethod
    @abstractmethod
    def save_tokens_mapping(path: Path, tokens_list: DocumentTokensMapping) -> None:
        ...

    @staticmethod
    @abstractmethod
    def load_tokens_mapping(path: Path) -> DocumentTokensMapping:
        ...


@dataclass
class TensorTokenizer(ABC):
    """Base class to tokenize abstracts into a tensor of token ids."""

    documents_info: DocumentsInfo

    @abstractmethod
    def tokenize(self) -> DocumentsTokensTensorMapping:
        ...

    @staticmethod
    @abstractmethod
    def save_tokens_mapping(path: Path, tokens_tensor: DocumentsTokensTensorMapping) -> None:
        ...

    @staticmethod
    @abstractmethod
    def load_tokens_mapping(path: Path) -> DocumentsTokensTensorMapping:
        ...


@dataclass
class SpacyTokenizer(ListTokenizer):
    """Tokenize abstracts using spacy into a list of string tokens."""

    documents_info: DocumentsInfo
    spacy_model: Language
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    remove_non_alphanumeric: bool = False

    def to_spacy_doc(self, document: str) -> Doc:
        """Converts a single abstract into a spacy document."""
        return self.spacy_model(document)

    def clean_spacy_doc(
        self,
        spacy_doc: Doc,
    ) -> DocumentTokens:
        """
        Cleans a single spacy document by removing stopwords, punctuation, and
        non-alphanumeric tokens.
        """
        clean_tokens = []
        # use for loop instead of list comprehension to allow disabling of individual filters
        for token in spacy_doc:
            if self.remove_stopwords and token.is_stop:
                continue

            if self.remove_punctuation and token.is_punct:
                continue

            if self.remove_non_alphanumeric and not token.is_alpha:
                continue

            clean_tokens.append(token.text)

        return clean_tokens

    def clean_document(self, document: DocumentString) -> DocumentTokens:
        """Tokenizes and cleans a single abstract."""
        spacy_doc = self.to_spacy_doc(document)
        return self.clean_spacy_doc(spacy_doc)

    def tokenize(self) -> DocumentTokensMapping:
        """
        Tokenizes and cleans multiple abstracts. Gnereates a mapping of document ids to
        tokens.
        """
        tokenized_abstracts_mapping = {}

        with setup_progress_bar() as progress_bar:
            for document_id, abstract in progress_bar.track(
                zip(self.documents_info.document_ids, self.documents_info.abstracts),
                total=len(self.documents_info.document_ids),
            ):
                tokenized_abstracts_mapping[document_id] = self.clean_document(abstract)

        return tokenized_abstracts_mapping

    def to_strings(self) -> DocumentStringMapping:
        """
        Generates a mapping of document ids to strings from raw abstracts. Required
        since `TfidfVectorizer` expects a list of strings, not a list of lists of
        strings.
        """
        return {document_id: " ".join(tokens) for document_id, tokens in self.tokenize().items()}

    @staticmethod
    def strings_from_tokens(
        tokens_mapping: DocumentTokensMapping,
    ) -> DocumentStringMapping:
        """
        Converts a mapping of document ids to tokens (list of string) into a mapping of
        document ids to strings. Required since `TfidfVectorizer` expects a list of
        strings, not a list of lists of strings.
        """
        return {document_id: " ".join(tokens) for document_id, tokens in tokens_mapping.items()}

    @staticmethod
    def save_tokens_mapping(path: Path, tokens_mapping: DocumentTokensMapping) -> None:
        """Save a mapping of document ids to tokens to a pickle file."""
        save_object_to_pickle(tokens_mapping, path)

    @staticmethod
    def load_tokens_mapping(path: Path) -> DocumentTokensMapping:
        """Load a mapping of document ids to tokens from a pickle file."""
        return load_object_from_pickle(path)  # type: ignore


@dataclass
class BERTTokenizer(TensorTokenizer):
    """Tokenize abstracts using BERT into a tensor of token ids."""

    documents_info: DocumentsInfo
    bert_tokenizer: BertTokenizerFast

    def tokenize(self) -> DocumentsTokensTensorMapping:
        """
        Tokenizes multiple abstracts into token ids. Generates a mapping of document ids
        to token ids.
        """
        tokenized_abstracts = self.bert_tokenizer(
            self.documents_info.abstracts,
            # BERT takes 512 dimensional tensors as input
            max_length=512,
            # truncate longer documents down to 512 tokens
            truncation=True,
            # pad shorter documents up to 512 tokens
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        return dict(zip(self.documents_info.document_ids, tokenized_abstracts))  # type: ignore

    @staticmethod
    def save_tokens_mapping(
        path: Path, tokens_tensor_mapping: DocumentsTokensTensorMapping
    ) -> None:
        """Save a mapping of document ids to token ids to a pytorch file."""
        torch.save(tokens_tensor_mapping, path)

    @staticmethod
    def load_tokens_mapping(path: Path) -> DocumentsTokensTensorMapping:
        """Load a mapping of document ids to token ids from a pytorch file."""
        return torch.load(path)  # type: ignore
