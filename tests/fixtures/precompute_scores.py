import pytest

from readnext.evaluation.scoring import (
    precompute_co_citations,
    precompute_co_citations_polars,
    precompute_co_references,
    precompute_co_references_polars,
    precompute_cosine_similarities,
    precompute_cosine_similarities_polars,
)
from readnext.utils.aliases import DocumentsFrame, EmbeddingsFrame, ScoresFrame


@pytest.fixture
def precomputed_co_citations(test_documents_frame: DocumentsFrame) -> ScoresFrame:
    return precompute_co_citations(test_documents_frame.head(10))


@pytest.fixture
def precomputed_co_citations_polars(test_documents_frame: DocumentsFrame) -> ScoresFrame:
    return precompute_co_citations_polars(test_documents_frame.lazy().head(10))


@pytest.fixture
def precomputed_co_references(test_documents_frame: DocumentsFrame) -> ScoresFrame:
    return precompute_co_references(test_documents_frame.head(10))


@pytest.fixture
def precomputed_co_references_polars(test_documents_frame: DocumentsFrame) -> ScoresFrame:
    return precompute_co_references_polars(test_documents_frame.lazy().head(10))


@pytest.fixture
def precomputed_cosine_similarities(test_tfidf_embeddings_frame: EmbeddingsFrame) -> ScoresFrame:
    return precompute_cosine_similarities(test_tfidf_embeddings_frame.head(10))


@pytest.fixture
def precomputed_cosine_similarities_polars(
    test_tfidf_embeddings_frame: EmbeddingsFrame,
) -> ScoresFrame:
    return precompute_cosine_similarities_polars(test_tfidf_embeddings_frame.lazy().head(10))
