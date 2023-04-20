from dataclasses import dataclass
from typing import Protocol, overload

import pandas as pd
import torch
from spacy.language import Language
from spacy.tokens.doc import Doc
from transformers import BertTokenizerFast
from typing_extensions import Self

from readnext.modeling.utils.rich_progress_bars import setup_progress_bar


@dataclass
class DocumentInfo:
    document_id: int
    title: str
    abstract: str


@dataclass
class DocumentsInfo:
    documents_info: list[DocumentInfo]

    def __post_init__(self) -> None:
        self.document_ids = [document_info.document_id for document_info in self.documents_info]
        self.titles = [document_info.title for document_info in self.documents_info]
        self.abstracts = [document_info.abstract for document_info in self.documents_info]

    def __len__(self) -> int:
        return len(self.documents_info)

    @overload
    def __getitem__(self, index: int) -> DocumentInfo:
        ...

    @overload
    def __getitem__(self, index: slice) -> Self:
        ...

    def __getitem__(self, index: int | slice) -> DocumentInfo | Self:
        # return single document info for integer index
        if isinstance(index, int):
            return self.documents_info[index]
        # return list of document infos for slice index
        return self.__class__(self.documents_info[index])


@dataclass
class DocumentStatistics:
    similarity: float
    document_info: DocumentInfo


def documents_info_from_df(df: pd.DataFrame) -> DocumentsInfo:
    document_ids = df["document_id"].tolist()
    titles = df["title"].tolist()
    abstracts = df["abstract"].tolist()

    return DocumentsInfo(
        [
            DocumentInfo(document_id, title, abstract)
            for document_id, title, abstract in zip(document_ids, titles, abstracts)
        ]
    )


@dataclass
class DocumentPreprocessor(Protocol):
    documents_info: DocumentsInfo


@dataclass
class SpacyPreprocessor:
    documents_info: DocumentsInfo
    spacy_model: Language
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    remove_non_alphanumeric: bool = False

    def to_spacy_doc(self, document: str) -> Doc:
        return self.spacy_model(document)

    def preprocess_spacy_doc(
        self,
        spacy_doc: Doc,
    ) -> list[str]:
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

    def clean_document(self, document: str) -> list[str]:
        spacy_doc = self.to_spacy_doc(document)
        return self.preprocess_spacy_doc(spacy_doc)

    def tokenize(self) -> list[list[str]]:
        """Preprocesses all abstracts of input documents and tokenizes them."""
        progress_bar = setup_progress_bar()
        tokenized_abstracts = []

        with progress_bar:
            for abstract in progress_bar.track(self.documents_info.abstracts):
                tokenized_abstracts.append(self.clean_document(abstract))

        return tokenized_abstracts

    # `TfidfVectorizer` expects a list of strings, not a list of lists of strings
    def to_strings(self) -> list[str]:
        return [" ".join(tokens) for tokens in self.tokenize()]

    def strings_from_tokens(self, tokens_list: list[list[str]]) -> list[str]:
        return [" ".join(tokens) for tokens in tokens_list]


@dataclass
class BERTPreprocessor:
    documents_info: DocumentsInfo
    bert_tokenizer: BertTokenizerFast

    def as_token_ids(self) -> torch.Tensor:
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
