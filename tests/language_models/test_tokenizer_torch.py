import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import (
    BERTTokenizer,
    LongformerTokenizer,
    Tokens,
    TorchTokenizer,
)
from readnext.utils.aliases import DocumentsFrame

torch_tokenizer_fixtures = [lazy_fixture("bert_tokenizer"), lazy_fixture("longformer_tokenizer")]

tokenizer_expected_tokens_pairs = [
    (lazy_fixture("bert_tokenizer"), lazy_fixture("bert_expected_tokens")),
    (lazy_fixture("longformer_tokenizer"), lazy_fixture("longformer_expected_tokens")),
]


@pytest.mark.parametrize(("tokenizer", "expected_tokens"), tokenizer_expected_tokens_pairs)
def test_tokenize_single_document(
    tokenizer: TorchTokenizer, toy_abstract: str, expected_tokens: Tokens
) -> None:
    token_ids = tokenizer.tokenize_into_ids(toy_abstract)

    assert isinstance(token_ids, list)
    assert all(isinstance(token_id, int) for token_id in token_ids)

    tokens = tokenizer.token_ids_to_tokens(token_ids)

    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)
    assert tokens == expected_tokens

    # check that `token_ids_to_tokens` and `tokens_to_token_ids` are inverses
    assert tokenizer.tokens_to_token_ids(tokenizer.token_ids_to_tokens(token_ids)) == token_ids
    assert tokenizer.token_ids_to_tokens(tokenizer.tokens_to_token_ids(tokens)) == tokens


@pytest.mark.parametrize("tokenizer", torch_tokenizer_fixtures)
def test_tokenize(tokenizer: TorchTokenizer, test_documents_frame: DocumentsFrame) -> None:
    token_ids_frame = tokenizer.tokenize(test_documents_frame.head(3))

    assert isinstance(token_ids_frame, pl.DataFrame)
    assert token_ids_frame.width == 2
    assert token_ids_frame.columns == ["d3_document_id", "token_ids"]
    assert token_ids_frame.dtypes == [pl.Int64, pl.List(pl.Int64)]

    tokens_frame = tokenizer.token_ids_frame_to_tokens_frame(token_ids_frame)
    assert isinstance(tokens_frame, pl.DataFrame)
    assert tokens_frame.width == 2
    assert tokens_frame.columns == ["d3_document_id", "tokens"]
    assert tokens_frame.dtypes == [pl.Int64, pl.List(pl.Utf8)]

    # check that `token_ids_frame_to_tokens_frame` and `tokens_frame_to_token_ids_frame`
    # are inverses
    assert_frame_equal(
        tokenizer.token_ids_frame_to_tokens_frame(
            tokenizer.tokens_frame_to_token_ids_frame(tokens_frame)
        ),
        tokens_frame,
    )
    assert_frame_equal(
        tokenizer.tokens_frame_to_token_ids_frame(
            tokenizer.token_ids_frame_to_tokens_frame(token_ids_frame)
        ),
        token_ids_frame,
    )


def test_tokenize_empty_abstract_bert(bert_tokenizer: BERTTokenizer) -> None:
    abstract = ""
    token_ids = bert_tokenizer.tokenize_into_ids(abstract)
    tokens = bert_tokenizer.token_ids_to_tokens(token_ids)
    assert tokens == ["[CLS]", "[SEP]"]


def test_tokenize_empty_abstract_longformer(longformer_tokenizer: LongformerTokenizer) -> None:
    abstract = ""
    token_ids = longformer_tokenizer.tokenize_into_ids(abstract)
    tokens = longformer_tokenizer.token_ids_to_tokens(token_ids)
    assert tokens == ["<s>", "</s>"]
