from readnext.modeling.language_models import bm25


def test_bm25_equal_frequency() -> None:
    document_tokens = ["a", "b"]
    document_corpus = [["a"], ["b"], ["a", "b"]]
    embedding = bm25(document_tokens, document_corpus)

    assert len(embedding) == 2
    # The embedding should be the same for both terms since they occur equally often in
    # the document and in the corpus
    assert embedding[0] == embedding[1]


def test_bm25_unequal_frequency() -> None:
    document_tokens = ["a", "b"]
    document_corpus = [["a"], ["b"], ["a", "a"]]
    embedding = bm25(document_tokens, document_corpus)

    assert len(embedding) == 2
    # The embedding should be zero for both terms since they do not occur in the corpus
    assert embedding[0] != embedding[1]
    # document frequency of "a" is twice as high as document frequency of "b", i.e.
    # inverse document frequency of "a" is half of the inverse document frequency of "b"
    assert embedding[1] == 2 * embedding[0]


def test_bm25_zero_frequency() -> None:
    document_tokens = ["a", "b"]
    document_corpus = [["a"], ["b"], ["a", "a", "c"]]
    embedding = bm25(document_tokens, document_corpus)

    assert len(embedding) == 3
    # embedding for terms in vocabulary but not in document is zero
    assert embedding[2] == 0


def test_bm25_unseen_token() -> None:
    document_tokens = ["a", "b", "c"]
    document_corpus = [["a"], ["b"]]
    embedding = bm25(document_tokens, document_corpus)

    # unseen tokens that are not in the vocabulary do not impact the embedding dimension
    assert len(embedding) == 2
    assert embedding[0] == embedding[1]
