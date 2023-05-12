from typing import cast

import pytest
from transformers import LongformerTokenizerFast

from readnext.config import ModelVersions
from readnext.modeling import DocumentsInfo
from readnext.modeling.language_models import LongformerTokenizer, TokenIds, Tokens


@pytest.fixture(scope="module")
def longformer_tokenizer_transformers() -> LongformerTokenizerFast:
    return LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer,
        do_lower_case=True,
        clean_text=True,
    )


@pytest.fixture(scope="module")
def tokenized_abstract() -> Tokens:
    return [
        "<s>",
        "Ċ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "ĠAbstract",
        "Ġ1",
        ":",
        "ĠThis",
        "Ġis",
        "Ġan",
        "Ġexample",
        "Ġabstract",
        "Ġwith",
        "Ġvarious",
        "Ġcharacters",
        "!",
        "ĠIt",
        "Ċ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġcontains",
        "Ġnumbers",
        "Ġ1",
        ",",
        "Ġ2",
        ",",
        "Ġ3",
        "Ġand",
        "Ġspecial",
        "Ġcharacters",
        "Ġlike",
        "Ġ@",
        ",",
        "Ġ#",
        ",",
        "Ġ$",
        ".",
        "Ċ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "</s>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
    ]


@pytest.fixture(scope="module")
def longformer_tokenizer(
    documents_info: DocumentsInfo, longformer_tokenizer_transformers: LongformerTokenizerFast
) -> LongformerTokenizer:
    return LongformerTokenizer(documents_info, longformer_tokenizer_transformers)


def test_tokenize(longformer_tokenizer: LongformerTokenizer) -> None:
    token_ids_mapping = longformer_tokenizer.tokenize()
    assert isinstance(token_ids_mapping, dict)

    assert all(isinstance(key, int) for key in token_ids_mapping)
    assert all(isinstance(value, list) for value in token_ids_mapping.values())
    assert all(isinstance(token, int) for value in token_ids_mapping.values() for token in value)

    assert list(token_ids_mapping.keys()) == [1, 2, 3]


def test_ids_to_tokens(longformer_tokenizer: LongformerTokenizer) -> None:
    # Prepare some token ids
    assert longformer_tokenizer.tensor_tokenizer.bos_token_id is not None
    assert longformer_tokenizer.tensor_tokenizer.eos_token_id is not None
    assert longformer_tokenizer.tensor_tokenizer.pad_token_id is not None

    tokens_ids: TokenIds = [
        longformer_tokenizer.tensor_tokenizer.bos_token_id,  # <s>
        cast(int, longformer_tokenizer.tensor_tokenizer.convert_tokens_to_ids("Ċ")),
        cast(int, longformer_tokenizer.tensor_tokenizer.convert_tokens_to_ids("Abstract")),
        longformer_tokenizer.tensor_tokenizer.eos_token_id,  # </s>
        longformer_tokenizer.tensor_tokenizer.pad_token_id,  # <pad>
    ]

    # Expected output
    expected_tokens: Tokens = [
        "<s>",
        "Ċ",
        "Abstract",
        "</s>",
        "<pad>",
    ]

    # Test functionality and types
    actual_tokens: Tokens = longformer_tokenizer.ids_to_tokens(tokens_ids)
    assert actual_tokens == expected_tokens
    assert all(isinstance(token, str) for token in actual_tokens)

    # Test empty list
    assert longformer_tokenizer.ids_to_tokens([]) == []

    # Test single-item list
    single_token_id: TokenIds = [longformer_tokenizer.tensor_tokenizer.bos_token_id]
    assert longformer_tokenizer.ids_to_tokens(single_token_id) == ["<s>"]


def test_id_mapping_to_tokens_mapping(
    longformer_tokenizer: LongformerTokenizer, tokenized_abstract: Tokens
) -> None:
    token_ids_mapping = longformer_tokenizer.tokenize()
    tokens_mapping = longformer_tokenizer.id_mapping_to_tokens_mapping(token_ids_mapping)
    assert isinstance(tokens_mapping, dict)

    assert all(isinstance(key, int) for key in tokens_mapping)
    assert all(isinstance(value, list) for value in tokens_mapping.values())
    assert all(isinstance(token, str) for value in tokens_mapping.values() for token in value)

    assert list(tokens_mapping.keys()) == [1, 2, 3]
    # check first example abstract
    first_abstract_tokens = tokens_mapping[1]
    assert len(first_abstract_tokens) == 107
    assert first_abstract_tokens == tokenized_abstract
