from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from spacy.language import Language
from spacy.tokens.doc import Doc

from readnext.modeling.document_info import DocumentsInfo
from readnext.utils import (
    Tokens,
    TokensMapping,
    load_object_from_pickle,
    setup_progress_bar,
    write_object_to_pickle,
)


@dataclass
class ListTokenizer(ABC):
    """Base class to tokenize abstracts into a list of string tokens."""

    documents_info: DocumentsInfo

    @abstractmethod
    def tokenize_single_document(self, document: str) -> Tokens:
        ...

    @abstractmethod
    def tokenize(self) -> TokensMapping:
        ...

    @staticmethod
    def save_tokens_mapping(path: Path, tokens_mapping: TokensMapping) -> None:
        """Save a mapping of document ids to tokens to a pickle file."""
        write_object_to_pickle(tokens_mapping, path)

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
    text_processing_steps: TextProcessingSteps = TextProcessingSteps()  # noqa: RUF009

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

    def tokenize_single_document(self, document: str) -> Tokens:
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
            for d3_document_id, abstract in progress_bar.track(
                zip(self.documents_info.d3_document_ids, self.documents_info.abstracts),
                total=len(self.documents_info.d3_document_ids),
                description=f"{self.__class__.__name__}:",
            ):
                tokenized_abstracts_mapping[d3_document_id] = self.tokenize_single_document(
                    abstract
                )

        return tokenized_abstracts_mapping
