import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import TokenIdsFrame, TokensFrame

bert_based_token_ids_frames = [
    lazy_fixture("test_bert_token_ids_frame"),
    lazy_fixture("test_scibert_token_ids_frame"),
]

token_ids_frames = [
    *bert_based_token_ids_frames,
    lazy_fixture("test_longformer_token_ids_frame"),
]

tokens_frames = [lazy_fixture("test_spacy_tokens_frame")]


@pytest.mark.parametrize("tokens_frame", tokens_frames)
def test_tokens_frame(tokens_frame: TokensFrame) -> None:
    assert isinstance(tokens_frame, pl.DataFrame)

    assert tokens_frame.shape[1] == 2
    assert tokens_frame.columns == ["d3_document_id", "tokens"]
    assert tokens_frame.dtypes == [pl.Int64, pl.List(pl.Utf8)]

    # check that all tokens are non-empty
    assert all(len(tokens) > 0 for tokens in tokens_frame["tokens"])


@pytest.mark.parametrize("token_ids_frame", token_ids_frames)
def test_token_ids_frame(token_ids_frame: TokenIdsFrame) -> None:
    assert isinstance(token_ids_frame, pl.DataFrame)

    assert token_ids_frame.shape[1] == 2
    assert token_ids_frame.columns == ["d3_document_id", "token_ids"]
    assert token_ids_frame.dtypes == [pl.Int64, pl.List(pl.Int64)]

    # check that all token ids are non-empty
    assert all(len(token_ids) > 0 for token_ids in token_ids_frame["token_ids"])


def test_bert_token_ids_frame(test_bert_token_ids_frame: TokenIdsFrame) -> None:
    single_abstract_token_ids = (
        test_bert_token_ids_frame.filter(pl.col("d3_document_id") == 13756489)
        .select("token_ids")
        .item()
    )

    # only padding token_ids from index 242 onwards
    assert sum(token_id != 0 for token_id in single_abstract_token_ids[:242])
    assert all(token_id == 0 for token_id in single_abstract_token_ids[242:])


def test_scibert_token_ids_frame(test_scibert_token_ids_frame: TokenIdsFrame) -> None:
    single_abstract_token_ids = (
        test_scibert_token_ids_frame.filter(pl.col("d3_document_id") == 13756489)
        .select("token_ids")
        .item()
    )

    # only padding token_ids from index 217 onwards
    assert sum(token_id != 0 for token_id in single_abstract_token_ids[:217])
    assert all(token_id == 0 for token_id in single_abstract_token_ids[217:])


def test_longformer_token_ids_frame(test_longformer_token_ids_frame: TokenIdsFrame) -> None:
    single_abstract_token_ids = (
        test_longformer_token_ids_frame.filter(pl.col("d3_document_id") == 13756489)
        .select("token_ids")
        .item()
    )

    # only padding token_ids from index 261 onwards
    assert all(token_id == 1 for token_id in single_abstract_token_ids[261:])
