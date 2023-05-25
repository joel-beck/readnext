import pandas as pd
import pytest

from readnext.modeling import DocumentInfo, DocumentScore, DocumentsInfo
from readnext.utils import Tokens


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


@pytest.fixture(scope="session")
def document_tokens() -> Tokens:
    return ["a", "b", "c", "a", "b", "c", "d", "d", "d"]


@pytest.fixture(scope="session")
def document_corpus() -> list[Tokens]:
    return [
        ["a", "b", "c", "d", "d", "d"],
        ["a", "b", "b", "c", "c", "c", "d"],
        ["a", "a", "a", "b", "c", "d"],
    ]


@pytest.fixture(scope="session")
def documents_info() -> DocumentsInfo:
    return DocumentsInfo(
        [
            DocumentInfo(
                d3_document_id=1,
                title="Title 1",
                author="Author 1",
                abstract="""
                Abstract 1: This is an example abstract with various characters! It
                contains numbers 1, 2, 3 and special characters like @, #, $.
                """,
            ),
            DocumentInfo(
                d3_document_id=2,
                title="Title 2",
                author="Author 2",
                abstract="""
                Abstract 2: Another example abstract, including upper-case letters and a
                few stopwords such as 'the', 'and', 'in'.
                """,
            ),
            DocumentInfo(
                d3_document_id=3,
                title="Title 3",
                author="Author 3",
                abstract="""
                Abstract 3: A third example abstract with a mix of lower-case and
                UPPER-CASE letters, as well as some punctuation: (brackets) and {curly
                braces}.
                """,
            ),
        ]
    )
