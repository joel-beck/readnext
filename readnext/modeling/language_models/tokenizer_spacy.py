from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl
from spacy.language import Language
from spacy.tokens.doc import Doc
from tqdm import tqdm

from readnext.utils.aliases import DocumentsFrame, Tokens, TokensFrame
from readnext.utils.progress_bar import tqdm_progress_bar_wrapper


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
class ListTokenizer(ABC):
    """Base class to tokenize abstracts into a list of string tokens."""

    @abstractmethod
    def tokenize_single_document(self, document: str) -> Tokens:
        ...

    def tokenize(self, documents_frame: DocumentsFrame) -> TokensFrame:
        """
        Tokenizes and cleans multiple abstracts. Generates a polars data frame with two
        columns named `d3_document_id` and `tokens`.
        """

        abstracts_frame = documents_frame.select(["d3_document_id", "abstract"])

        with tqdm(total=len(abstracts_frame)) as progress_bar:
            tokens_frame = abstracts_frame.with_columns(
                tokens=pl.col("abstract").apply(
                    tqdm_progress_bar_wrapper(progress_bar, self.tokenize_single_document)
                )
            )

        return tokens_frame.drop("abstract")


@dataclass
class Tokenizer(ABC):
    """Base class to tokenize abstracts into a list of string tokens."""

    @abstractmethod
    def tokenize_single_document(self, document: str) -> Tokens:
        ...

    def tokenize(self, documents_frame: DocumentsFrame) -> TokensFrame:
        """
        Tokenizes and cleans multiple abstracts. Generates a polars data frame with two
        columns named `d3_document_id` and `tokens`.
        """

        with tqdm(total=len(documents_frame)) as progress_bar:
            tokens_frame = documents_frame.with_columns(
                tokens=pl.col("abstract").apply(
                    tqdm_progress_bar_wrapper(progress_bar, self.tokenize_single_document)
                )
            )

        return tokens_frame.select(["d3_document_id", "tokens"])


@dataclass
class SpacyTokenizer(Tokenizer):
    """Tokenize abstracts using spacy into a list of string tokens."""

    spacy_model: Language
    text_processing_steps: TextProcessingSteps = TextProcessingSteps()  # noqa: RUF009

    def to_spacy_doc(self, document: str) -> Doc:
        """Converts a single abstract into a spacy document."""
        return self.spacy_model(document)

    def clean_spacy_doc(
        self,
        spacy_doc: Doc,
        stopwords: Container | None = None,
    ) -> Tokens:
        """
        Cleans a single spacy document by removing stopwords, punctuation, and
        non-alphanumeric tokens.

        Accepts a custom set or list of stopwords to be removed. If no stopwords are
        passed, the default stopword list of the spacy model is used.
        """
        stopwords = self.spacy_model.Defaults.stop_words if stopwords is None else stopwords

        clean_tokens = []
        # use for loop instead of list comprehension to allow disabling of individual filters
        for token in spacy_doc:
            # initialize token_text such that lemmatisation and lowercasing can be
            # applied independently of each other
            token_text = token.text

            # NOTE: Lemmatization and lowercasing must come first since removal of
            # stopwords only works on lowercase tokens!
            if self.text_processing_steps.lemmatize:
                token_text = token.lemma_

            if self.text_processing_steps.to_lowercase:
                token_text = token_text.lower()

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

            if self.text_processing_steps.remove_stopwords and token_text in stopwords:
                continue

            clean_tokens.append(token_text)

        return clean_tokens

    def tokenize_single_document(self, document: str) -> Tokens:
        """Tokenizes and cleans a single abstract."""
        spacy_doc = self.to_spacy_doc(document)
        return self.clean_spacy_doc(spacy_doc)
