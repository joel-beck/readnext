"""
Implements BM25+ algorithm as proposed by (Lv & Zhai, 2011) and described in (Trotman et
al., 2014).
"""

from collections.abc import Sequence

import numpy as np

from readnext.modeling.language_models.tfidf import df, learn_vocabulary, tf
from readnext.modeling.language_models.tokenizer_spacy import Tokens


def bm25_tf(
    term: str,
    document_tokens: Tokens,
    document_corpus: Sequence[Tokens],
    k: float,
    b: float,
    delta: float,
) -> float:
    """
    BM25+ Term Frequency:
    [(k + 1) * tf(t, d)] / [k * (1 - b + b * (len(d) / avgdl)) + tf(t, d)] + delta
    where k and b are free parameters, len(d) is the length of the document in tokens,
    and avgdl is the average document length in tokens.

    Computes the BM25 term frequency for a single term.
    """
    numerator = (k + 1) * tf(term, document_tokens)

    average_document_length = np.mean(
        [len(document_tokens) for document_tokens in document_corpus]
    ).astype(float)

    denominator = (k * (1 - b + b * (len(document_tokens) / average_document_length))) + tf(
        term, document_tokens
    )

    return numerator / denominator + delta


def bm25_idf(term: str, document_corpus: Sequence[Tokens]) -> float:
    """
    BM25+ Inverse Document Frequency: log[(N+1) / df(t)]
    where n is the total number of documents in the corpus and df(t) is the number of
    documents in the corpus that contain the term t.
    """
    numerator = len(document_corpus) + 1
    denominator = df(term, document_corpus)

    return np.log(numerator / denominator)


def bm25_single_term(
    term: str,
    document_tokens: Tokens,
    document_corpus: Sequence[Tokens],
    k: float,
    b: float,
    delta: float,
) -> float:
    """
    BM25+: BM25+ Term Frequency * BM25+ Inverse Document Frequency

    Computes the BM25+ vector for a single term.
    """
    return bm25_tf(term, document_tokens, document_corpus, k, b, delta) * bm25_idf(
        term, document_corpus
    )


def bm25(
    document_tokens: Tokens,
    document_corpus: Sequence[Tokens],
    k: float = 1.5,
    b: float = 0.75,
    delta: float = 1.0,
) -> np.ndarray:
    """
    Default values are taken from the `rank_bm25` package:
    https://github.com/dorianbrown/rank_bm25/blob/990470ebbe6b28c18216fd1a8b18fe7446237dd6/rank_bm25.py#L176

    Computes the BM25+ vector for a single document.
    """
    corpus_vocabulary = learn_vocabulary(document_corpus)

    return np.array(
        [
            bm25_single_term(term, document_tokens, document_corpus, k, b, delta)
            if term in document_tokens
            else 0
            for term in corpus_vocabulary
        ]
    )
