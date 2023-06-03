import numpy as np
import polars as pl
import pytest

from readnext.evaluation.scoring import (
    precompute_co_citations,
    precompute_co_references,
    precompute_cosine_similarities,
)
from readnext.utils import EmbeddingsFrame


@pytest.fixture
def document_embeddings_df() -> EmbeddingsFrame:
    data = {
        "d3_document_id": [1, 2, 3, 4],
        "embedding": [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]),
        ],
    }
    return pl.DataFrame(data)


@pytest.fixture
def co_citation_analysis_scores(test_documents_data: pl.DataFrame) -> pl.DataFrame:
    return precompute_co_citations(test_documents_data.head(10))


@pytest.fixture
def bibliographic_coupling_scores(test_documents_data: pl.DataFrame) -> pl.DataFrame:
    return precompute_co_references(test_documents_data.head(10))


@pytest.fixture
def tfidf_embeddings(test_tfidf_embeddings: pl.DataFrame) -> pl.DataFrame:
    return precompute_cosine_similarities(test_tfidf_embeddings.head(10))
