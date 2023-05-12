from typing import cast

import pytest
from transformers import BertTokenizerFast

from readnext.config import ModelVersions
from readnext.modeling import DocumentsInfo
from readnext.modeling.language_models import BERTTokenizer, TokenIds, Tokens


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
def tokenized_abstracts() -> list[Tokens]:
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


def test_ids_to_tokens(bert_tokenizer: BERTTokenizer) -> None:
    # Prepare some token ids
    assert bert_tokenizer.tensor_tokenizer.cls_token_id is not None
    assert bert_tokenizer.tensor_tokenizer.sep_token_id is not None
    assert bert_tokenizer.tensor_tokenizer.pad_token_id is not None

    tokens_ids: TokenIds = [
        bert_tokenizer.tensor_tokenizer.cls_token_id,  # [CLS]
        cast(int, bert_tokenizer.tensor_tokenizer.convert_tokens_to_ids("abstract")),
        cast(int, bert_tokenizer.tensor_tokenizer.convert_tokens_to_ids("1")),
        bert_tokenizer.tensor_tokenizer.sep_token_id,  # [SEP]
        bert_tokenizer.tensor_tokenizer.pad_token_id,  # [PAD]
    ]

    # Expected output
    expected_tokens: Tokens = [
        "[CLS]",
        "abstract",
        "1",
        "[SEP]",
        "[PAD]",
    ]

    # Test functionality and types
    actual_tokens: Tokens = bert_tokenizer.ids_to_tokens(tokens_ids)
    assert actual_tokens == expected_tokens
    assert all(isinstance(token, str) for token in actual_tokens)

    # Test empty list
    assert bert_tokenizer.ids_to_tokens([]) == []

    # Test single-item list
    single_token_id: TokenIds = [bert_tokenizer.tensor_tokenizer.cls_token_id]
    assert bert_tokenizer.ids_to_tokens(single_token_id) == ["[CLS]"]


def test_id_mapping_to_tokens_mapping(
    bert_tokenizer: BERTTokenizer, tokenized_abstracts: list[Tokens]
) -> None:
    token_ids_mapping = bert_tokenizer.tokenize()
    tokens_mapping = bert_tokenizer.id_mapping_to_tokens_mapping(token_ids_mapping)
    assert isinstance(tokens_mapping, dict)

    assert all(isinstance(key, int) for key in tokens_mapping)
    assert all(isinstance(value, list) for value in tokens_mapping.values())
    assert all(isinstance(token, str) for value in tokens_mapping.values() for token in value)

    assert list(tokens_mapping.keys()) == [1, 2, 3]
    assert list(tokens_mapping.values()) == tokenized_abstracts
