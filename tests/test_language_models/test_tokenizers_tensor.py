from typing import cast

import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import (
    BERTTokenizer,
    LongformerTokenizer,
    TensorTokenizer,
    TokenIds,
    Tokens,
)


def test_tokenize_bert(bert_tokenizer: BERTTokenizer) -> None:
    token_ids_mapping = bert_tokenizer.tokenize()
    assert isinstance(token_ids_mapping, dict)

    assert all(isinstance(key, int) for key in token_ids_mapping)
    assert all(isinstance(value, list) for value in token_ids_mapping.values())
    assert all(isinstance(token, int) for value in token_ids_mapping.values() for token in value)

    assert list(token_ids_mapping.keys()) == [1, 2, 3]


def test_ids_to_tokens_bert(bert_tokenizer: BERTTokenizer) -> None:
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


def test_tokenize_longformer(longformer_tokenizer: LongformerTokenizer) -> None:
    token_ids_mapping = longformer_tokenizer.tokenize()
    assert isinstance(token_ids_mapping, dict)

    assert all(isinstance(key, int) for key in token_ids_mapping)
    assert all(isinstance(value, list) for value in token_ids_mapping.values())
    assert all(isinstance(token, int) for value in token_ids_mapping.values() for token in value)

    assert list(token_ids_mapping.keys()) == [1, 2, 3]


def test_ids_to_tokens_longformer(longformer_tokenizer: LongformerTokenizer) -> None:
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


tokenizers = ["bert_tokenizer", "longformer_tokenizer"]


@pytest.mark.parametrize(
    "tokenizer",
    lazy_fixture(tokenizers),
)
def test_id_mapping_to_tokens_mapping(tokenizer: TensorTokenizer) -> None:
    token_ids_mapping = tokenizer.tokenize()
    tokens_mapping = tokenizer.id_mapping_to_tokens_mapping(token_ids_mapping)
    assert isinstance(tokens_mapping, dict)

    assert all(isinstance(key, int) for key in tokens_mapping)
    assert all(isinstance(value, list) for value in tokens_mapping.values())
    assert all(isinstance(token, str) for value in tokens_mapping.values() for token in value)

    # tokenizer has taken fixture `documents_info` as input which consists of 3 abstracts
    assert list(tokens_mapping.keys()) == [1, 2, 3]


def test_id_to_tokens_mapping_bert(
    bert_tokenizer: BERTTokenizer, bert_exptected_tokenized_abstract: Tokens
) -> None:
    token_ids_mapping = bert_tokenizer.tokenize()
    tokens_mapping = bert_tokenizer.id_mapping_to_tokens_mapping(token_ids_mapping)
    actual_tokenized_abstract = tokens_mapping[1]

    assert actual_tokenized_abstract == bert_exptected_tokenized_abstract
    assert len(actual_tokenized_abstract) == 40
    assert actual_tokenized_abstract[0] == "[CLS]"
    assert "[SEP]" in actual_tokenized_abstract
    assert actual_tokenized_abstract[-1] == "[PAD]"


def test_id_to_tokens_mapping_longformer(
    longformer_tokenizer: LongformerTokenizer, longformer_expected_tokenized_abstract: Tokens
) -> None:
    token_ids_mapping = longformer_tokenizer.tokenize()
    tokens_mapping = longformer_tokenizer.id_mapping_to_tokens_mapping(token_ids_mapping)
    actual_tokenized_abstract = tokens_mapping[1]

    assert actual_tokenized_abstract == longformer_expected_tokenized_abstract
    assert len(actual_tokenized_abstract) == 107
    assert actual_tokenized_abstract[0] == "<s>"
    assert "</s>" in actual_tokenized_abstract
    assert actual_tokenized_abstract[-1] == "<pad>"
