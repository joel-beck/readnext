import polars as pl
import pytest

from readnext.modeling import DocumentInfo
from readnext.utils.aliases import Tokens


@pytest.fixture
def citation_models_features_frame() -> pl.DataFrame:
    data = {
        "publication_date": ["2020-01-01", "2019-01-01", None],
        "citationcount_document": [50, 100, 75],
        "citationcount_author": [1000, 2000, 3000],
    }
    return pl.DataFrame(data)


@pytest.fixture
def extended_citation_models_features_frame() -> pl.DataFrame:
    data = {
        "publication_date": ["2020-01-01", "2019-01-01", None, "2018-01-01", "2017-01-01"],
        "citationcount_document": [50, 100, 75, 50, 100],
        "citationcount_author": [1000, 2000, 3000, 1000, 2000],
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_document_info() -> DocumentInfo:
    return DocumentInfo(
        d3_document_id=1,
        title="Sample Paper",
        author="John Doe",
        arxiv_labels=["cs.AI", "cs.CL"],
        semanticscholar_url=(
            "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
        ),
        arxiv_url="https://arxiv.org/abs/2106.01572",
        abstract="This is a sample paper.",
    )


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
