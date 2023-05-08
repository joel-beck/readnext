import pytest
import spacy
from spacy.language import Language
from spacy.tokens.doc import Doc

from readnext.config import ModelVersions
from readnext.modeling import DocumentInfo, DocumentsInfo
from readnext.modeling.language_models import SpacyTokenizer


@pytest.fixture(scope="module")
def spacy_model() -> Language:
    return spacy.load(ModelVersions.spacy)


@pytest.fixture(scope="module")
def spacy_tokenizer(documents_info: DocumentsInfo, spacy_model: Language) -> SpacyTokenizer:
    return SpacyTokenizer(documents_info, spacy_model)


@pytest.fixture(scope="module")
def tokenized_abstracts() -> list[list[str]]:
    return [
        [
            "abstract",
            "example",
            "abstract",
            "character",
            "contain",
            "number",
            "special",
            "character",
            "like",
        ],
        ["abstract", "example", "abstract", "include", "upper", "case", "letter", "stopword"],
        [
            "abstract",
            "example",
            "abstract",
            "mix",
            "low",
            "case",
            "upper",
            "case",
            "letter",
            "punctuation",
            "bracket",
            "curly",
            "brace",
        ],
    ]


@pytest.fixture(scope="module")
def tokenized_abstracts_strings() -> list[str]:
    return [
        "abstract example abstract character contain number special character like",
        "abstract example abstract include upper case letter stopword",
        "abstract example abstract mix low case upper case letter punctuation bracket curly brace",  # noqa: E501
    ]


def test_to_spacy_doc(spacy_tokenizer: SpacyTokenizer, documents_info: DocumentsInfo) -> None:
    assert all(
        isinstance(spacy_tokenizer.to_spacy_doc(abstract), Doc)
        for abstract in documents_info.abstracts
    )


def test_clean_spacy_doc(
    spacy_tokenizer: SpacyTokenizer,
    documents_info: DocumentsInfo,
    tokenized_abstracts: list[list[str]],
) -> None:
    docs = [spacy_tokenizer.to_spacy_doc(abstract) for abstract in documents_info.abstracts]

    docs_clean = [spacy_tokenizer.clean_spacy_doc(doc) for doc in docs]

    assert all(isinstance(doc, list) for doc in docs_clean)
    assert all(isinstance(token, str) for doc in docs_clean for token in doc)

    assert docs_clean == tokenized_abstracts


def test_tokenize(spacy_tokenizer: SpacyTokenizer, tokenized_abstracts: list[list[str]]) -> None:
    tokens_mapping = spacy_tokenizer.tokenize()
    assert isinstance(tokens_mapping, dict)

    assert all(isinstance(key, int) for key in tokens_mapping)
    assert all(isinstance(value, list) for value in tokens_mapping.values())
    assert all(isinstance(token, str) for value in tokens_mapping.values() for token in value)

    assert list(tokens_mapping.keys()) == [1, 2, 3]
    assert list(tokens_mapping.values()) == tokenized_abstracts


def test_to_strings(
    spacy_tokenizer: SpacyTokenizer, tokenized_abstracts_strings: list[str]
) -> None:
    string_mapping = spacy_tokenizer.to_strings()
    assert isinstance(string_mapping, dict)

    assert all(isinstance(key, int) for key in string_mapping)
    assert all(isinstance(value, str) for value in string_mapping.values())

    assert list(string_mapping.keys()) == [1, 2, 3]
    assert list(string_mapping.values()) == tokenized_abstracts_strings


def test_strings_from_tokens(
    tokenized_abstracts: list[list[str]], tokenized_abstracts_strings: list[str]
) -> None:
    tokens_mapping = dict(enumerate(tokenized_abstracts, start=1))
    string_mapping_from_tokens_mapping = SpacyTokenizer.string_mapping_from_tokens_mapping(
        tokens_mapping
    )

    assert isinstance(string_mapping_from_tokens_mapping, dict)

    assert all(isinstance(key, int) for key in string_mapping_from_tokens_mapping)
    assert all(isinstance(value, str) for value in string_mapping_from_tokens_mapping.values())

    assert list(string_mapping_from_tokens_mapping.keys()) == [1, 2, 3]
    assert list(string_mapping_from_tokens_mapping.values()) == tokenized_abstracts_strings


def test_tokenize_empty_abstract(spacy_model: Language) -> None:
    documents_info = DocumentsInfo(
        [
            DocumentInfo(document_id=1, title="Empty paper", abstract=""),
        ]
    )

    spacy_tokenizer = SpacyTokenizer(documents_info, spacy_model)
    tokenized_abstracts_mapping = spacy_tokenizer.tokenize()
    assert tokenized_abstracts_mapping[1] == []


def test_tokenize_stopwords(spacy_tokenizer: SpacyTokenizer) -> None:
    stopwords = set(spacy_tokenizer.spacy_model.Defaults.stop_words)
    tokenized_abstracts_mapping = spacy_tokenizer.tokenize()

    for tokens in tokenized_abstracts_mapping.values():
        assert all(token not in stopwords for token in tokens)


def test_tokenize_punctuation(spacy_tokenizer: SpacyTokenizer) -> None:
    tokenized_abstracts_mapping = spacy_tokenizer.tokenize()

    for tokens in tokenized_abstracts_mapping.values():
        assert all(token.isalnum() for token in tokens)


def test_tokenize_lowercase(spacy_tokenizer: SpacyTokenizer) -> None:
    tokenized_abstracts_mapping = spacy_tokenizer.tokenize()

    for tokens in tokenized_abstracts_mapping.values():
        assert all(token.islower() for token in tokens)
