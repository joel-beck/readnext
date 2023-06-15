import polars as pl
import pytest

from readnext.evaluation.scoring import (
    precompute_co_citations_polars,
    precompute_co_references_polars,
    precompute_cosine_similarities_polars,
)
from readnext.utils.aliases import DocumentsFrame, EmbeddingsFrame


@pytest.fixture
def co_citation_analysis_scores(test_documents_frame: DocumentsFrame) -> pl.DataFrame:
    return precompute_co_citations_polars(test_documents_frame.lazy().head(10))


@pytest.fixture
def bibliographic_coupling_scores(test_documents_frame: DocumentsFrame) -> pl.DataFrame:
    return precompute_co_references_polars(test_documents_frame.lazy().head(10))


@pytest.fixture
def tfidf_embeddings(test_tfidf_embeddings: EmbeddingsFrame) -> pl.DataFrame:
    return precompute_cosine_similarities_polars(test_tfidf_embeddings.lazy().head(10))
