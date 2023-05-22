import pandas as pd
import pytest
import numpy as np

from readnext.evaluation.scoring import (
    precompute_co_citations,
    precompute_co_references,
    precompute_cosine_similarities,
)


@pytest.fixture
def document_embeddings_df() -> pd.DataFrame:
    data = {
        "document_id": [1, 2, 3, 4],
        "embedding": [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]),
        ],
    }
    return pd.DataFrame(data).set_index("document_id")


@pytest.fixture
def co_citation_analysis_scores(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> pd.DataFrame:
    return precompute_co_citations(test_documents_authors_labels_citations_most_cited.head(10))


@pytest.fixture
def bibliographic_coupling_scores(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> pd.DataFrame:
    return precompute_co_references(test_documents_authors_labels_citations_most_cited.head(10))


@pytest.fixture
def tfidf_embeddings(test_tfidf_embeddings_most_cited: pd.DataFrame) -> pd.DataFrame:
    return precompute_cosine_similarities(test_tfidf_embeddings_most_cited.head(10))
