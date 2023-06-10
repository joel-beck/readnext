import numpy as np
import pytest

from readnext.modeling.language_models import (
    Tokens,
    df,
    idf,
    learn_vocabulary,
    tf,
    tfidf,
    tfidf_single_term,
)


def test_tf(toy_document_tokens: Tokens) -> None:
    assert tf("a", toy_document_tokens) == 2 / 9
    assert tf("b", toy_document_tokens) == 2 / 9
    assert tf("c", toy_document_tokens) == 2 / 9
    assert tf("d", toy_document_tokens) == 3 / 9
    # Edge case: term does not exist in document
    assert tf("z", toy_document_tokens) == 0
    # Edge case: empty document
    with pytest.raises(ZeroDivisionError):
        tf("a", [])


def test_df(toy_document_corpus: list[Tokens]) -> None:
    assert df("a", toy_document_corpus) == 3  # Appears in all documents
    assert df("b", toy_document_corpus) == 3  # Appears in all documents
    assert df("c", toy_document_corpus) == 3  # Appears in all documents
    assert df("d", toy_document_corpus) == 3  # Appears in all documents
    assert df("e", toy_document_corpus) == 0  # Does not appear in any document


def test_idf(toy_document_corpus: list[Tokens]) -> None:
    assert idf("a", toy_document_corpus) == np.log(
        (1 + 3) / (1 + 3) + 1
    )  # Appears in all documents
    assert idf("b", toy_document_corpus) == np.log(
        (1 + 3) / (1 + 3) + 1
    )  # Appears in all documents
    assert idf("c", toy_document_corpus) == np.log(
        (1 + 3) / (1 + 3) + 1
    )  # Appears in all documents
    assert idf("d", toy_document_corpus) == np.log(
        (1 + 3) / (1 + 3) + 1
    )  # Appears in all documents
    # Does not appear in any document
    assert idf("e", toy_document_corpus) == np.log((1 + 3) / (1 + 0) + 1)
    # Edge case: term does not exist in corpus
    np.testing.assert_almost_equal(idf("z", toy_document_corpus), 1.6094379124341003)
    # Edge case: empty corpus does NOT evaluate to zero
    np.testing.assert_almost_equal(idf("a", []), 0.6931471805599453)


def test_learn_vocabulary(toy_document_corpus: list[Tokens]) -> None:
    # Common case
    assert learn_vocabulary(toy_document_corpus) == ["a", "b", "c", "d"]

    # Test empty corpus
    assert learn_vocabulary([]) == []

    # Test corpus with empty documents
    assert learn_vocabulary([[], [], []]) == []


def test_tfidf_single_term(toy_document_tokens: Tokens, toy_document_corpus: list[Tokens]) -> None:
    # Common case
    assert tfidf_single_term("d", toy_document_tokens, toy_document_corpus) == tf(
        "d", toy_document_tokens
    ) * idf("d", toy_document_corpus)
    # Edge case: term does not exist in document
    assert tfidf_single_term("z", toy_document_tokens, toy_document_corpus) == 0
    # Edge case: empty document
    with pytest.raises(ZeroDivisionError):
        tfidf_single_term("d", Tokens([]), toy_document_corpus)


def test_tfidf(toy_document_tokens: Tokens, toy_document_corpus: list[Tokens]) -> None:
    # Common case
    expected_tfidf = np.array([0.1540327, 0.1540327, 0.1540327, 0.2310491])
    np.testing.assert_almost_equal(tfidf(toy_document_tokens, toy_document_corpus), expected_tfidf)
    # Edge case: term does not exist in document
    assert (tfidf(Tokens(["z"]), toy_document_corpus) == np.array([0, 0, 0, 0])).all()
    # Edge case: empty document
    assert (tfidf(Tokens([]), toy_document_corpus) == np.array([0, 0, 0, 0])).all()
