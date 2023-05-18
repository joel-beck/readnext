import pytest

from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
)


@pytest.fixture
def semanticscholar_url() -> str:
    return "https://www.semanticscholar.org/paper/5cc2cfb77c9643760f4e7e18"


@pytest.fixture
def semanticscholar_id() -> str:
    return "5cc2cfb77c9643760f4e7e18"


@pytest.fixture
def arxiv_url() -> str:
    return "https://arxiv.org/abs/2101.10000"


@pytest.fixture
def arxiv_id() -> str:
    return "2101.10000"


# test get_paper_id_from_semanticscholar_url function
def test_get_semanticscholar_id_from_semanticscholar_url(
    semanticscholar_url: str, semanticscholar_id: str
) -> None:
    assert (
        get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url) == semanticscholar_id
    )
    assert (
        get_semanticscholar_id_from_semanticscholar_url("https://www.semanticscholar.org/paper/")
        == ""
    )
    assert get_semanticscholar_id_from_semanticscholar_url("") == ""


# test get_semanticscholar_url_from_paper_id function
def test_get_semanticscholar_url_from_semanticscholar_id(
    semanticscholar_url: str, semanticscholar_id: str
) -> None:
    assert (
        get_semanticscholar_url_from_semanticscholar_id(semanticscholar_id) == semanticscholar_url
    )
    assert get_semanticscholar_url_from_semanticscholar_id(None) == ""


# test get_arxiv_id_from_arxiv_url function
def test_get_arxiv_id_from_arxiv_url(arxiv_url: str, arxiv_id: str) -> None:
    assert get_arxiv_id_from_arxiv_url(arxiv_url) == arxiv_id
    assert get_arxiv_id_from_arxiv_url("https://arxiv.org/abs/") == ""
    assert get_arxiv_id_from_arxiv_url("") == ""


# test get_arxiv_url_from_arxiv_id function
def test_get_arxiv_url_from_arxiv_id(arxiv_url: str, arxiv_id: str) -> None:
    assert get_arxiv_url_from_arxiv_id(arxiv_id) == arxiv_url
    assert get_arxiv_url_from_arxiv_id(None) == ""
