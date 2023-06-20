import numpy as np
import pytest

from readnext.modeling.language_models import (
    Tokens,
    bm25,
    bm25_idf,
    bm25_single_term,
    bm25_tf,
    df,
    learn_vocabulary,
    tf,
)


@pytest.mark.updated
def test_bm25_tf(toy_document_tokens: Tokens, toy_document_corpus: list[Tokens]) -> None:
    k = 1.5
    b = 0.75
    delta = 1.0

    assert bm25_tf("a", toy_document_tokens, toy_document_corpus, k, b, delta) == (
        ((k + 1) * tf("a", toy_document_tokens))
        / (
            k
            * (
                1
                - b
                + b
                * (len(toy_document_tokens) / np.mean([len(doc) for doc in toy_document_corpus]))
            )
            + tf("a", toy_document_tokens)
        )
        + delta
    )

    assert bm25_tf("b", toy_document_tokens, toy_document_corpus, k, b, delta) == (
        ((k + 1) * tf("b", toy_document_tokens))
        / (
            k
            * (
                1
                - b
                + b
                * (len(toy_document_tokens) / np.mean([len(doc) for doc in toy_document_corpus]))
            )
            + tf("b", toy_document_tokens)
        )
        + delta
    )

    assert bm25_tf("c", toy_document_tokens, toy_document_corpus, k, b, delta) == (
        ((k + 1) * tf("c", toy_document_tokens))
        / (
            k
            * (
                1
                - b
                + b
                * (len(toy_document_tokens) / np.mean([len(doc) for doc in toy_document_corpus]))
            )
            + tf("c", toy_document_tokens)
        )
        + delta
    )

    assert bm25_tf("d", toy_document_tokens, toy_document_corpus, k, b, delta) == (
        ((k + 1) * tf("d", toy_document_tokens))
        / (
            k
            * (
                1
                - b
                + b
                * (len(toy_document_tokens) / np.mean([len(doc) for doc in toy_document_corpus]))
            )
            + tf("d", toy_document_tokens)
        )
        + delta
    )


@pytest.mark.updated
def test_bm25_idf(toy_document_corpus: list[Tokens]) -> None:
    assert bm25_idf("a", toy_document_corpus) == np.log(
        (len(toy_document_corpus) + 1) / df("a", toy_document_corpus)
    )
    assert bm25_idf("b", toy_document_corpus) == np.log(
        (len(toy_document_corpus) + 1) / df("b", toy_document_corpus)
    )
    assert bm25_idf("c", toy_document_corpus) == np.log(
        (len(toy_document_corpus) + 1) / df("c", toy_document_corpus)
    )
    assert bm25_idf("d", toy_document_corpus) == np.log(
        (len(toy_document_corpus) + 1) / df("d", toy_document_corpus)
    )


@pytest.mark.updated
def test_bm25(toy_document_tokens: Tokens, toy_document_corpus: list[Tokens]) -> None:
    k = 1.5
    b = 0.75
    delta = 1.0
    corpus_vocabulary = learn_vocabulary(toy_document_corpus)
    expected_bm25 = np.array(
        [
            bm25_single_term(term, toy_document_tokens, toy_document_corpus, k, b, delta)
            if term in toy_document_tokens
            else 0
            for term in corpus_vocabulary
        ]
    )
    np.testing.assert_array_equal(
        bm25(toy_document_tokens, toy_document_corpus, k, b, delta), expected_bm25
    )

    # Test document with terms not in corpus
    toy_document_tokens = ["e", "f", "g"]
    expected_bm25 = np.array(
        [
            bm25_single_term(term, toy_document_tokens, toy_document_corpus, k, b, delta)
            if term in toy_document_tokens
            else 0
            for term in corpus_vocabulary
        ]
    )
    np.testing.assert_array_equal(
        bm25(toy_document_tokens, toy_document_corpus, k, b, delta), expected_bm25
    )

    # Test corpus with empty documents
    document_corpus_with_empty_docs: list[list[str]] = [[], ["a", "b"], [], ["c", "d", "e"], []]
    toy_document_tokens = ["a", "b", "c"]
    corpus_vocabulary = learn_vocabulary(document_corpus_with_empty_docs)
    expected_bm25 = np.array(
        [
            bm25_single_term(
                term, toy_document_tokens, document_corpus_with_empty_docs, k, b, delta
            )
            if term in toy_document_tokens
            else 0
            for term in corpus_vocabulary
        ]
    )
    np.testing.assert_array_equal(
        bm25(toy_document_tokens, document_corpus_with_empty_docs, k, b, delta), expected_bm25
    )
