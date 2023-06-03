import pytest

from readnext.modeling import DocumentInfo, DocumentScore


def test_document_info(sample_document_info: DocumentInfo) -> None:
    assert sample_document_info.d3_document_id == 1
    assert sample_document_info.title == "Sample Paper"
    assert sample_document_info.author == "John Doe"
    assert sample_document_info.arxiv_labels == ["cs.AI", "cs.CL"]
    assert sample_document_info.abstract == "This is a sample paper."

    str_representation = (
        "Document 1\n"
        "---------------------\n"
        "Title: Sample Paper\n"
        "Author: John Doe\n"
        "Arxiv Labels: ['cs.AI', 'cs.CL']"
    )
    assert str(sample_document_info) == str_representation


def test_document_score(
    sample_document_score: DocumentScore, sample_document_info: DocumentInfo
) -> None:
    assert sample_document_score.document_info == sample_document_info
    assert sample_document_score.score == 0.75


def test_document_info_defaults() -> None:
    document_info = DocumentInfo(d3_document_id=3)

    assert document_info.d3_document_id == 3
    assert document_info.title == ""
    assert document_info.author == ""
    assert document_info.arxiv_labels == []
    assert document_info.abstract == ""

    str_representation = "Document 3\n---------------------\nTitle: \nAuthor: \nArxiv Labels: []"
    assert str(document_info) == str_representation


def test_kw_only_initialization_document_info() -> None:
    with pytest.raises(TypeError):
        DocumentInfo(
            -1,  # type: ignore
            "Title",
            "Author",
            ["cs.AI", "cs.CL"],
            "Abstract",
        )


def test_kw_only_initialization_document_score() -> None:
    with pytest.raises(TypeError):
        DocumentScore(
            DocumentInfo(
                d3_document_id=-1,  # type: ignore
                title="Title",
                author="Author",
                arxiv_labels=["cs.AI", "cs.CL"],
                abstract="Abstract",
            ),
            0.75,
        )
