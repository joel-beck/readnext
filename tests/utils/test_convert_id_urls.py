from readnext.utils.convert_id_urls import (
    get_arxiv_id_from_arxiv_url,
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
)


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


def test_get_semanticscholar_url_from_semanticscholar_id(
    semanticscholar_url: str, semanticscholar_id: str
) -> None:
    assert (
        get_semanticscholar_url_from_semanticscholar_id(semanticscholar_id) == semanticscholar_url
    )
    assert get_semanticscholar_url_from_semanticscholar_id(None) == ""


def test_get_arxiv_id_from_arxiv_url(arxiv_url: str, arxiv_id: str) -> None:
    assert get_arxiv_id_from_arxiv_url(arxiv_url) == arxiv_id
    assert get_arxiv_id_from_arxiv_url("https://arxiv.org/abs/") == ""
    assert get_arxiv_id_from_arxiv_url("") == ""


def test_get_arxiv_url_from_arxiv_id(arxiv_url: str, arxiv_id: str) -> None:
    assert get_arxiv_url_from_arxiv_id(arxiv_id) == arxiv_url
    assert get_arxiv_url_from_arxiv_id(None) == ""
