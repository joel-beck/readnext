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


def test_tf(document_tokens: Tokens) -> None:
    assert tf("a", document_tokens) == 2 / 9
    assert tf("b", document_tokens) == 2 / 9
    assert tf("c", document_tokens) == 2 / 9
    assert tf("d", document_tokens) == 3 / 9
    # Edge case: term does not exist in document
    assert tf("z", document_tokens) == 0
    # Edge case: empty document
    with pytest.raises(ZeroDivisionError):
        tf("a", [])


def test_df(document_corpus: list[Tokens]) -> None:
    assert df("a", document_corpus) == 3  # Appears in all documents
    assert df("b", document_corpus) == 3  # Appears in all documents
    assert df("c", document_corpus) == 3  # Appears in all documents
    assert df("d", document_corpus) == 3  # Appears in all documents
    assert df("e", document_corpus) == 0  # Does not appear in any document


def test_idf(document_corpus: list[Tokens]) -> None:
    assert idf("a", document_corpus) == np.log((1 + 3) / (1 + 3) + 1)  # Appears in all documents
    assert idf("b", document_corpus) == np.log((1 + 3) / (1 + 3) + 1)  # Appears in all documents
    assert idf("c", document_corpus) == np.log((1 + 3) / (1 + 3) + 1)  # Appears in all documents
    assert idf("d", document_corpus) == np.log((1 + 3) / (1 + 3) + 1)  # Appears in all documents
    # Does not appear in any document
    assert idf("e", document_corpus) == np.log((1 + 3) / (1 + 0) + 1)
    # Edge case: term does not exist in corpus
    np.testing.assert_almost_equal(idf("z", document_corpus), 1.6094379124341003)
    # Edge case: empty corpus does NOT evaluate to zero
    np.testing.assert_almost_equal(idf("a", []), 0.6931471805599453)


def test_learn_vocabulary(document_corpus: list[Tokens]) -> None:
    # Common case
    assert learn_vocabulary(document_corpus) == ["a", "b", "c", "d"]

    # Test empty corpus
    assert learn_vocabulary([]) == []

    # Test corpus with empty documents
    assert learn_vocabulary([[], [], []]) == []


def test_tfidf_single_term(document_tokens: Tokens, document_corpus: list[Tokens]) -> None:
    # Common case
    assert tfidf_single_term("d", document_tokens, document_corpus) == tf(
        "d", document_tokens
    ) * idf("d", document_corpus)
    # Edge case: term does not exist in document
    assert tfidf_single_term("z", document_tokens, document_corpus) == 0
    # Edge case: empty document
    with pytest.raises(ZeroDivisionError):
        tfidf_single_term("d", Tokens([]), document_corpus)


def test_tfidf(document_tokens: Tokens, document_corpus: list[Tokens]) -> None:
    # Common case
    expected_tfidf = np.array([0.1540327, 0.1540327, 0.1540327, 0.2310491])
    np.testing.assert_almost_equal(tfidf(document_tokens, document_corpus), expected_tfidf)
    # Edge case: term does not exist in document
    assert (tfidf(Tokens(["z"]), document_corpus) == np.array([0, 0, 0, 0])).all()
    # Edge case: empty document
    assert (tfidf(Tokens([]), document_corpus) == np.array([0, 0, 0, 0])).all()
