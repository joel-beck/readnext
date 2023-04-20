from dataclasses import dataclass
from typing import Protocol, TypeAlias

import torch
from spacy.language import Language
from spacy.tokens.doc import Doc
from transformers import BertTokenizerFast

from readnext.modeling.language_models import DocumentsInfo
from readnext.modeling.utils.rich_progress_bars import setup_progress_bar

DocumentTokens: TypeAlias = list[str]
DocumentsTokensList: TypeAlias = list[DocumentTokens]
DocumentsTokensString: TypeAlias = list[str]
DocumentsTokensTensor: TypeAlias = torch.Tensor


@dataclass
class ListTokenizer(Protocol):
    documents_info: DocumentsInfo

    def tokenize(self) -> DocumentsTokensList:
        ...


@dataclass
class TensorTokenizer(Protocol):
    documents_info: DocumentsInfo

    def tokenize(self) -> DocumentsTokensTensor:
        ...


@dataclass
class SpacyTokenizer:
    """Implements `ListTokenizer` Protocol"""

    documents_info: DocumentsInfo
    spacy_model: Language
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    remove_non_alphanumeric: bool = False

    def to_spacy_doc(self, document: str) -> Doc:
        return self.spacy_model(document)

    def clean_spacy_doc(
        self,
        spacy_doc: Doc,
    ) -> DocumentTokens:
        """Cleans a single spacy document."""
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

    def clean_document(self, document: str) -> DocumentTokens:
        """Converts and cleans a single abstract."""
        spacy_doc = self.to_spacy_doc(document)
        return self.clean_spacy_doc(spacy_doc)

    def tokenize(self) -> DocumentsTokensList:
        """Cleans and tokenizes multiple abstracts."""
        progress_bar = setup_progress_bar()
        tokenized_abstracts = []

        with progress_bar:
            for abstract in progress_bar.track(self.documents_info.abstracts):
                tokenized_abstracts.append(self.clean_document(abstract))

        return tokenized_abstracts

    # `TfidfVectorizer` expects a list of strings, not a list of lists of strings
    def to_strings(self) -> DocumentsTokensString:
        return [" ".join(tokens) for tokens in self.tokenize()]

    @staticmethod
    def strings_from_tokens(tokens_list: DocumentsTokensList) -> DocumentsTokensString:
        return [" ".join(tokens) for tokens in tokens_list]


@dataclass
class BERTTokenizer:
    """Implement `TensorTokenizer` Protocol"""

    documents_info: DocumentsInfo
    bert_tokenizer: BertTokenizerFast

    def tokenize(self) -> DocumentsTokensTensor:
        return self.bert_tokenizer(
            self.documents_info.abstracts,
            # BERT takes 512 dimensional tensors as input
            max_length=512,
            # truncate longer documents down to 512 tokens
            truncation=True,
            # pad shorter documents up to 512 tokens
            padding=True,
            return_tensors="pt",
        )[
            "input_ids"
        ]  # type: ignore
