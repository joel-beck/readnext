import numpy as np

from readnext.modeling.language_models import Tokens, df, idf, tf, tfidf


def test_tf(document_tokens: Tokens) -> None:
    assert tf("a", document_tokens) == 2 / 9
    assert tf("b", document_tokens) == 2 / 9
    assert tf("c", document_tokens) == 2 / 9
    assert tf("d", document_tokens) == 3 / 9
    assert tf("e", document_tokens) == 0.0  # Test term not in document


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


def test_tfidf(document_tokens: Tokens, document_corpus: list[Tokens]) -> None:
    expected_tfidf = np.array(
        [tf(term, document_tokens) * idf(term, document_corpus) for term in document_tokens]
    )
    np.testing.assert_array_equal(tfidf(document_tokens, document_corpus), expected_tfidf)

    # Test document with terms not in corpus
    document_tokens = ["e", "f", "g"]
    expected_tfidf = np.array(
        [tf(term, document_tokens) * idf(term, document_corpus) for term in document_tokens]
    )
    np.testing.assert_array_equal(tfidf(document_tokens, document_corpus), expected_tfidf)
