import pytest
from transformers import BertTokenizerFast

from readnext.config import ModelVersions
from readnext.modeling import DocumentsInfo
from readnext.modeling.language_models import BERTTokenizer


@pytest.fixture(scope="module")
def bert_tokenizer_transformers() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(
        ModelVersions.bert,
        do_lower_case=True,
        clean_text=True,
    )


@pytest.fixture(scope="module")
def bert_tokenizer(
    documents_info: DocumentsInfo, bert_tokenizer_transformers: BertTokenizerFast
) -> BERTTokenizer:
    return BERTTokenizer(documents_info, bert_tokenizer_transformers)


@pytest.fixture(scope="module")
def tokenized_abstracts() -> list[list[str]]:
    return [
        [
            "[CLS]",
            "abstract",
            "1",
            ":",
            "this",
            "is",
            "an",
            "example",
            "abstract",
            "with",
            "various",
            "characters",
            "!",
            "it",
            "contains",
            "numbers",
            "1",
            ",",
            "2",
            ",",
            "3",
            "and",
            "special",
            "characters",
            "like",
            "@",
            ",",
            "#",
            ",",
            "$",
            ".",
            "[SEP]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
        ],
        [
            "[CLS]",
            "abstract",
            "2",
            ":",
            "another",
            "example",
            "abstract",
            ",",
            "including",
            "upper",
            "-",
            "case",
            "letters",
            "and",
            "a",
            "few",
            "stop",
            "##words",
            "such",
            "as",
            "'",
            "the",
            "'",
            ",",
            "'",
            "and",
            "'",
            ",",
            "'",
            "in",
            "'",
            ".",
            "[SEP]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
            "[PAD]",
        ],
        [
            "[CLS]",
            "abstract",
            "3",
            ":",
            "a",
            "third",
            "example",
            "abstract",
            "with",
            "a",
            "mix",
            "of",
            "lower",
            "-",
            "case",
            "and",
            "upper",
            "-",
            "case",
            "letters",
            ",",
            "as",
            "well",
            "as",
            "some",
            "pun",
            "##ct",
            "##uation",
            ":",
            "(",
            "brackets",
            ")",
            "and",
            "{",
            "curly",
            "brace",
            "##s",
            "}",
            ".",
            "[SEP]",
        ],
    ]


def test_tokenize(bert_tokenizer: BERTTokenizer) -> None:
    token_ids_mapping = bert_tokenizer.tokenize()
    assert isinstance(token_ids_mapping, dict)

    assert all(isinstance(key, int) for key in token_ids_mapping)
    assert all(isinstance(value, list) for value in token_ids_mapping.values())
    assert all(isinstance(token, int) for value in token_ids_mapping.values() for token in value)

    assert list(token_ids_mapping.keys()) == [1, 2, 3]


def test_id_mapping_to_tokens_mapping(
    bert_tokenizer: BERTTokenizer, tokenized_abstracts: list[list[str]]
) -> None:
    token_ids_mapping = bert_tokenizer.tokenize()
    tokens_mapping = bert_tokenizer.id_mapping_to_tokens_mapping(token_ids_mapping)
    assert isinstance(tokens_mapping, dict)

    assert all(isinstance(key, int) for key in tokens_mapping)
    assert all(isinstance(value, list) for value in tokens_mapping.values())
    assert all(isinstance(token, str) for value in tokens_mapping.values() for token in value)

    assert list(tokens_mapping.keys()) == [1, 2, 3]
    assert list(tokens_mapping.values()) == tokenized_abstracts
