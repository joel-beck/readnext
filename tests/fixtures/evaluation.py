import polars as pl
import pytest

from readnext.evaluation.scoring import (
    precompute_co_citations,
    precompute_co_references,
    precompute_cosine_similarities,
)
from readnext.utils.aliases import DocumentsFrame, EmbeddingsFrame


@pytest.fixture
def co_citation_analysis_scores(test_documents_frame: DocumentsFrame) -> pl.DataFrame:
    return precompute_co_citations(test_documents_frame.head(10))


@pytest.fixture
def bibliographic_coupling_scores(test_documents_frame: DocumentsFrame) -> pl.DataFrame:
    return precompute_co_references(test_documents_frame.head(10))


@pytest.fixture
def tfidf_embeddings(test_tfidf_embeddings: EmbeddingsFrame) -> pl.DataFrame:
    return precompute_cosine_similarities(test_tfidf_embeddings.head(10))
