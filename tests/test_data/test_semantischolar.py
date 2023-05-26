import pytest

from readnext.data import SemanticscholarRequest, SemanticScholarResponse


def test_post_init(semanticscholar_request: SemanticscholarRequest) -> None:
    assert isinstance(semanticscholar_request, SemanticscholarRequest)
    assert isinstance(semanticscholar_request.semanticscholar_api_key, str)
    assert isinstance(semanticscholar_request.request_headers, dict)
    assert "x-api-key" in semanticscholar_request.request_headers
    assert (
        semanticscholar_request.request_headers["x-api-key"]
        == semanticscholar_request.semanticscholar_api_key
    )


def test_get_response_from_request_url(
    semanticscholar_request: SemanticscholarRequest,
    semanticscholar_response: SemanticScholarResponse,
    mock_get_response_from_request: None,  # noqa: ARG001
) -> None:
    test_url = "https://api.semanticscholar.org/graph/v1/paper/TestID"
    response = semanticscholar_request.get_response_from_request_url(test_url)
    assert isinstance(response, SemanticScholarResponse)
    assert response == semanticscholar_response


def test_from_semanticscholar_id(
    semanticscholar_request: SemanticscholarRequest,
    semanticscholar_response: SemanticScholarResponse,
    mock_get_response_from_request: None,  # noqa: ARG001
) -> None:
    semanticscholar_id = "TestID"
    response = semanticscholar_request.from_semanticscholar_id(semanticscholar_id)
    assert isinstance(response, SemanticScholarResponse)
    assert response == semanticscholar_response


def test_from_arxiv_id(
    semanticscholar_request: SemanticscholarRequest,
    semanticscholar_response: SemanticScholarResponse,
    mock_get_response_from_request: None,  # noqa: ARG001
) -> None:
    arxiv_id = "TestID"
    response = semanticscholar_request.from_arxiv_id(arxiv_id)
    assert isinstance(response, SemanticScholarResponse)
    assert response == semanticscholar_response


def test_from_arxiv_url(
    semanticscholar_request: SemanticscholarRequest,
    semanticscholar_response: SemanticScholarResponse,
    mock_get_response_from_request: None,  # noqa: ARG001
) -> None:
    arxiv_url = "https://arxiv.org/abs/TestID"
    response = semanticscholar_request.from_arxiv_url(arxiv_url)
    assert isinstance(response, SemanticScholarResponse)
    assert response == semanticscholar_response


def test_kw_only_initialization_semanticscholar() -> None:
    with pytest.raises(TypeError):
        SemanticScholarResponse(
            "SemantischscholarID",  # type: ignore
            "ArxviID",
            "Title",
            "Abstract",
            [],
            [],
        )
