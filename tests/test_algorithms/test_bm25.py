import numpy as np

from readnext.modeling.language_models import Tokens, bm25, bm25_idf, bm25_tf, df, tf


def test_bm25_tf(document_tokens: Tokens, document_corpus: list[Tokens]) -> None:
    k = 1.5
    b = 0.75
    delta = 1.0

    assert bm25_tf("a", document_tokens, document_corpus, k, b, delta) == (
        ((k + 1) * tf("a", document_tokens))
        / (
            k
            * (1 - b + b * (len(document_tokens) / np.mean([len(doc) for doc in document_corpus])))
            + tf("a", document_tokens)
        )
        + delta
    )

    assert bm25_tf("b", document_tokens, document_corpus, k, b, delta) == (
        ((k + 1) * tf("b", document_tokens))
        / (
            k
            * (1 - b + b * (len(document_tokens) / np.mean([len(doc) for doc in document_corpus])))
            + tf("b", document_tokens)
        )
        + delta
    )

    assert bm25_tf("c", document_tokens, document_corpus, k, b, delta) == (
        ((k + 1) * tf("c", document_tokens))
        / (
            k
            * (1 - b + b * (len(document_tokens) / np.mean([len(doc) for doc in document_corpus])))
            + tf("c", document_tokens)
        )
        + delta
    )

    assert bm25_tf("d", document_tokens, document_corpus, k, b, delta) == (
        ((k + 1) * tf("d", document_tokens))
        / (
            k
            * (1 - b + b * (len(document_tokens) / np.mean([len(doc) for doc in document_corpus])))
            + tf("d", document_tokens)
        )
        + delta
    )


def test_bm25_idf(document_corpus: list[Tokens]) -> None:
    assert bm25_idf("a", document_corpus) == np.log(
        (len(document_corpus) + 1) / df("a", document_corpus)
    )
    assert bm25_idf("b", document_corpus) == np.log(
        (len(document_corpus) + 1) / df("b", document_corpus)
    )
    assert bm25_idf("c", document_corpus) == np.log(
        (len(document_corpus) + 1) / df("c", document_corpus)
    )
    assert bm25_idf("d", document_corpus) == np.log(
        (len(document_corpus) + 1) / df("d", document_corpus)
    )


def test_bm25(document_tokens: Tokens, document_corpus: list[Tokens]) -> None:
    k = 1.5
    b = 0.75
    delta = 1.0
    expected_bm25 = np.array(
        [
            bm25_tf(term, document_tokens, document_corpus, k, b, delta)
            * bm25_idf(term, document_corpus)
            for term in document_tokens
        ]
    )
    np.testing.assert_array_equal(
        bm25(document_tokens, document_corpus, k, b, delta), expected_bm25
    )

    # Test document with terms not in corpus
    document_tokens = ["e", "f", "g"]
    expected_bm25 = np.array(
        [
            bm25_tf(term, document_tokens, document_corpus, k, b, delta)
            * bm25_idf(term, document_corpus)
            for term in document_tokens
        ]
    )
    np.testing.assert_array_equal(
        bm25(document_tokens, document_corpus, k, b, delta), expected_bm25
    )
