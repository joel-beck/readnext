import polars as pl
from spacy.language import Language
from transformers import BertTokenizerFast, LongformerTokenizerFast


from readnext.modeling.language_models import SpacyTokenizer
from readnext.utils.aliases import Tokens


def test_tokenize_documents_frame(spacy_tokenizer: SpacyTokenizer) -> None:
    tokens_frame = spacy_tokenizer.tokenize()

    assert isinstance(tokens_frame, pl.DataFrame)

    assert tokens_frame.shape[1] == 2
    assert tokens_frame.columns == ["d3_document_id", "tokens"]

    assert tokens_frame["d3_document_id"].dtype == pl.Int64
    assert tokens_frame["tokens"].dtype == pl.List(pl.Utf8)

    # test that all tokens are alphanumeric characters
    assert all(token.isalnum() for tokens in tokens_frame["tokens"] for token in tokens)

    # test that all tokens are ascii characters
    assert all(token.isascii() for tokens in tokens_frame["tokens"] for token in tokens)

    # test that all digits are removed
    assert all(not token.isdigit() for tokens in tokens_frame["tokens"] for token in tokens)

    # test that all tokens are lowercased
    assert all(token.islower() for tokens in tokens_frame["tokens"] for token in tokens)

    # test that all whitespace is removed
    assert all(not token.isspace() for tokens in tokens_frame["tokens"] for token in tokens)

    # test that all stopwords are removed
    assert all(
        token not in spacy_tokenizer.spacy_model.Defaults.stop_words
        for tokens in tokens_frame["tokens"]
        for token in tokens
    )


def test_tokenize_toy_abstract(
    spacy_tokenizer: SpacyTokenizer, toy_abstract: str, spacy_expected_tokens: Tokens
) -> None:
    tokens = spacy_tokenizer.tokenize_single_document(toy_abstract)

    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)

    assert tokens == spacy_expected_tokens


def test_tokenize_empty_abstract(spacy_tokenizer: SpacyTokenizer) -> None:
    tokens = spacy_tokenizer.tokenize_single_document("")
    assert tokens == []
