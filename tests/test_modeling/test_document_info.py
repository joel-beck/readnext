import pytest

from readnext.modeling import DocumentInfo


def test_document_info(dummy_document_info: DocumentInfo) -> None:
    assert dummy_document_info.d3_document_id == 1
    assert dummy_document_info.title == "Sample Paper"
    assert dummy_document_info.author == "John Doe"
    assert dummy_document_info.arxiv_labels == ["cs.AI", "cs.CL"]
    assert dummy_document_info.semanticscholar_url == (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    assert dummy_document_info.arxiv_url == "https://arxiv.org/abs/2106.01572"
    assert dummy_document_info.abstract == "This is a sample paper."

    str_representation = (
        "Document 1\n"
        "---------------------\n"
        "Title: Sample Paper\n"
        "Author: John Doe\n"
        "Arxiv Labels: ['cs.AI', 'cs.CL']\n"
        "Semanticscholar URL: https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776\n"
        "Arxiv URL: https://arxiv.org/abs/2106.01572"
    )
    assert str(dummy_document_info) == str_representation


def test_document_info_defaults() -> None:
    document_info = DocumentInfo(d3_document_id=3)

    assert document_info.d3_document_id == 3
    assert document_info.title == ""
    assert document_info.author == ""
    assert document_info.arxiv_labels == []
    assert document_info.semanticscholar_url == ""
    assert document_info.arxiv_url == ""
    assert document_info.abstract == ""

    str_representation = (
        "Document 3\n"
        "---------------------\n"
        "Title: \n"
        "Author: \n"
        "Arxiv Labels: []\n"
        "Semanticscholar URL: \n"
        "Arxiv URL:"
    )
    assert str(document_info) == str_representation


def test_kw_only_initialization_document_info() -> None:
    with pytest.raises(TypeError):
        DocumentInfo(
            -1,  # type: ignore
            "Title",
            "Author",
            ["cs.AI", "cs.CL"],
            "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            "https://arxiv.org/abs/2106.01572",
            "Abstract",
        )
