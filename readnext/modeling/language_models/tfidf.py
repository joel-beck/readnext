"""
Implements the TF-IDF algorithm with the following specification:
- The term frequency is interpreted as the relative frequency of a term in a document
(not the absolute frequency/count as in `scikit-learn`).
- The inverse document frequency is computed as in `scikit-learn`:
https://`scikit-learn`.org/stable/modules/feature_extraction.html#tfidf-term-weighting
- The tfidf score is not normalized by the Euclidean norm as in `scikit-learn` since the
term frequency is already a relative quantity.
"""

from collections.abc import Sequence

import numpy as np

from readnext.modeling.language_models.tokenizer_spacy import Tokens


def tf(term: str, document_tokens: Tokens) -> float:
    """
    Term Frequency: tf(t, d) = count(t, d) / len(d)

    Commputes the term frequency for a single term.
    """
    return document_tokens.count(term) / len(document_tokens)


def df(term: str, document_corpus: Sequence[Tokens]) -> int:
    """
    Document Frequency: df(t) = count(d in corpus: t in d),
    i.e. the number of documents in the corpus that contain the term t.

    Computes the document frequency for a single term.
    """
    return np.sum([term in document_tokens for document_tokens in document_corpus])


def idf(term: str, document_corpus: Sequence[Tokens]) -> float:
    """
    Inverse Document Frequency: IDF(t) = log[(1+n) / (1+df(t)) + 1]
    where n is the total number of documents in the corpus and df(t) is the number of
    documents in the corpus that contain the term t.

    Computes the inverse document frequency for a single term.
    """
    numerator = 1 + len(document_corpus)
    denominator = 1 + df(term, document_corpus)

    return np.log((numerator / denominator) + 1)


def tfidf_single_term(
    term: str, document_tokens: Tokens, document_corpus: Sequence[Tokens]
) -> float:
    """
    TF-IDF: tf(t, d) * idf(t)

    Computes the TF-IDF score for a single term.
    """
    return tf(term, document_tokens) * idf(term, document_corpus)


def learn_vocabulary(document_corpus: Sequence[Tokens]) -> Tokens:
    """
    Returns a sorted list of all unique terms in the document corpus.
    """
    return sorted({term for document_tokens in document_corpus for term in document_tokens})


def tfidf(document_tokens: Tokens, document_corpus: Sequence[Tokens]) -> np.ndarray:
    """
    Computes the TF-IDF vector for a single document. The output is a sparse vector
    which length equals the vocabulary size of the document corpus.
    """
    corpus_vocabulary = learn_vocabulary(document_corpus)

    return np.array(
        [
            tfidf_single_term(term, document_tokens, document_corpus)
            if term in document_tokens
            else 0
            for term in corpus_vocabulary
        ]
    )
