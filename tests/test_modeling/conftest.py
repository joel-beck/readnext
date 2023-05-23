import pandas as pd
import pytest

from readnext.modeling import DocumentInfo, DocumentScore, DocumentsInfo


@pytest.fixture
def citation_models_features_frame() -> pd.DataFrame:
    data = {
        "publication_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2019-01-01"), None],
        "citationcount_document": [50, 100, 75],
        "citationcount_author": [1000, 2000, 3000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def extended_citation_models_features_frame() -> pd.DataFrame:
    data = {
        "publication_date": [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2019-01-01"),
            None,
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2017-01-01"),
        ],
        "citationcount_document": [50, 100, 75, 50, 100],
        "citationcount_author": [1000, 2000, 3000, 1000, 2000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_document_info() -> DocumentInfo:
    return DocumentInfo(
        d3_document_id=1,
        title="Sample Paper",
        author="John Doe",
        arxiv_labels=["cs.AI", "cs.CL"],
        abstract="This is a sample paper.",
    )


@pytest.fixture
def sample_documents_info(sample_document_info: DocumentInfo) -> DocumentsInfo:
    return DocumentsInfo(
        documents_info=[
            sample_document_info,
            DocumentInfo(
                d3_document_id=2,
                title="Another Sample Paper",
                author="Jane Doe",
                arxiv_labels=["cs.CV", "cs.LG"],
                abstract="This is another sample paper.",
            ),
        ]
    )


@pytest.fixture
def sample_document_score(sample_document_info: DocumentInfo) -> DocumentScore:
    return DocumentScore(document_info=sample_document_info, score=0.75)
