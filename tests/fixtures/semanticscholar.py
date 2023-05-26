import pytest
from pytest_mock import MockFixture

from readnext.data import SemanticScholarJson, SemanticscholarRequest, SemanticScholarResponse


@pytest.fixture
def semanticscholar_request() -> SemanticscholarRequest:
    return SemanticscholarRequest()


@pytest.fixture
def json_response() -> SemanticScholarJson:
    return {
        "paperId": "TestID",
        "title": "TestTitle",
        "abstract": "TestAbstract",
        "citations": [],
        "references": [],
        "externalIds": {"ArXiv": "ArxivID", "DBLP": None, "PubMedCentral": None},
    }


@pytest.fixture
def semanticscholar_response() -> SemanticScholarResponse:
    return SemanticScholarResponse(
        semanticscholar_id="TestID",
        arxiv_id="ArxivID",
        title="TestTitle",
        abstract="TestAbstract",
        citations=[],
        references=[],
    )


@pytest.fixture
def mock_get_response_from_request(
    mocker: MockFixture, semanticscholar_response: SemanticScholarResponse
) -> None:
    mocker.patch.object(
        SemanticscholarRequest, "get_response_from_request", return_value=semanticscholar_response
    )
