from dataclasses import asdict

import pandas as pd
import pytest

from readnext.modeling import DocumentInfo, DocumentScore, DocumentsInfo, documents_info_from_df


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


def test_documents_info(sample_documents_info: DocumentsInfo) -> None:
    assert len(sample_documents_info) == 2
    assert sample_documents_info.d3_document_ids == [1, 2]
    assert sample_documents_info.titles == ["Sample Paper", "Another Sample Paper"]
    assert sample_documents_info.abstracts == [
        "This is a sample paper.",
        "This is another sample paper.",
    ]

    # Test __getitem__ with integer index
    document_info = sample_documents_info[0]
    assert isinstance(document_info, DocumentInfo)
    assert document_info.d3_document_id == 1

    # Test __getitem__ with slice index
    sliced_documents_info = sample_documents_info[1:]
    assert isinstance(sliced_documents_info, DocumentsInfo)
    assert len(sliced_documents_info) == 1
    assert sliced_documents_info.d3_document_ids == [2]


def test_document_score(
    sample_document_score: DocumentScore, sample_document_info: DocumentInfo
) -> None:
    assert sample_document_score.document_info == sample_document_info
    assert sample_document_score.score == 0.75


def test_documents_info_from_df() -> None:
    # Create a sample dataframe
    data = {
        "document_id": [1, 2],
        "title": ["Sample Paper", "Another Sample Paper"],
        "abstract": ["This is a sample paper.", "This is another sample paper."],
    }
    df = pd.DataFrame(data)

    # Test the function
    documents_info = documents_info_from_df(df)
    assert isinstance(documents_info, DocumentsInfo)
    assert len(documents_info) == 2
    assert documents_info.d3_document_ids == [1, 2]
    assert documents_info.titles == ["Sample Paper", "Another Sample Paper"]
    assert documents_info.abstracts == ["This is a sample paper.", "This is another sample paper."]

    assert asdict(documents_info_from_df(df)[0]) == {
        "d3_document_id": 1,
        "title": "Sample Paper",
        "abstract": "This is a sample paper.",
        "author": "",
        "arxiv_labels": [],
    }


def test_document_info_defaults() -> None:
    document_info = DocumentInfo(d3_document_id=3)

    assert document_info.d3_document_id == 3
    assert document_info.title == ""
    assert document_info.author == ""
    assert document_info.arxiv_labels == []
    assert document_info.abstract == ""

    str_representation = "Document 3\n---------------------\nTitle: \nAuthor: \nArxiv Labels: []"
    assert str(document_info) == str_representation


def test_documents_info_empty() -> None:
    documents_info = DocumentsInfo(documents_info=[])

    assert len(documents_info) == 0
    assert documents_info.d3_document_ids == []
    assert documents_info.titles == []
    assert documents_info.abstracts == []

    with pytest.raises(IndexError):
        _ = documents_info[0]

    sliced_documents_info = documents_info[:]
    assert isinstance(sliced_documents_info, DocumentsInfo)
    assert len(sliced_documents_info) == 0
    assert sliced_documents_info.d3_document_ids == []


def test_documents_info_from_df_empty() -> None:
    df = pd.DataFrame(columns=["document_id", "title", "abstract"])
    documents_info = documents_info_from_df(df)

    assert isinstance(documents_info, DocumentsInfo)
    assert len(documents_info) == 0
    assert documents_info.d3_document_ids == []
    assert documents_info.titles == []
    assert documents_info.abstracts == []


def test_documents_info_from_df_partial_data() -> None:
    data = {
        "document_id": [1],
        "title": ["Sample Paper"],
    }
    df = pd.DataFrame(data)

    with pytest.raises(KeyError):
        _ = documents_info_from_df(df)

    # If the 'abstract' column is missing, the function should fail.
    data = {
        "document_id": [1],
        "abstract": ["This is a sample paper."],
    }
    df = pd.DataFrame(data)

    with pytest.raises(KeyError):
        _ = documents_info_from_df(df)


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
