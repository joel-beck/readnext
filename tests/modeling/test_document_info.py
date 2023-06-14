import dataclasses

import pytest

from readnext.inference import InferenceData
from readnext.modeling import DocumentInfo


def test_from_dummy(dummy_document_info: DocumentInfo) -> None:
    assert dummy_document_info.d3_document_id == 1
    assert dummy_document_info.title == "Sample Paper"
    assert dummy_document_info.author == "John Doe"
    assert dummy_document_info.publication_date == "2000-01-01"
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
        "Publication Date: 2000-01-01\n"
        "Arxiv Labels: ['cs.AI', 'cs.CL']\n"
        "Semanticscholar URL: https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776\n"
        "Arxiv URL: https://arxiv.org/abs/2106.01572"
    )
    assert str(dummy_document_info) == str_representation


def test_from_inference_data(inference_data: InferenceData) -> None:
    assert isinstance(inference_data.document_info, DocumentInfo)
    assert list(dataclasses.asdict(inference_data.document_info)) == [
        "d3_document_id",
        "title",
        "author",
        "publication_date",
        "arxiv_labels",
        "semanticscholar_url",
        "arxiv_url",
        "abstract",
    ]


def test_from_inference_data_seen(inference_data_seen: InferenceData) -> None:
    assert inference_data_seen.document_info.d3_document_id == 13756489
    assert inference_data_seen.document_info.title == "Attention Is All You Need"
    assert inference_data_seen.document_info.author == "Ashish Vaswani"
    assert inference_data_seen.document_info.publication_date == "2017-06-12"
    assert inference_data_seen.document_info.arxiv_labels == ["cs.CL", "cs.LG"]
    assert (
        inference_data_seen.document_info.semanticscholar_url
        == "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    assert inference_data_seen.document_info.arxiv_url == "https://arxiv.org/abs/1706.03762"
    assert len(inference_data_seen.document_info.abstract) > 0


def test_from_inference_data_unseen(inference_data_unseen: InferenceData) -> None:
    assert inference_data_unseen.document_info.d3_document_id == -1
    assert inference_data_unseen.document_info.title == "GPT-4 Technical Report"
    # author is not set for unseen papers
    assert inference_data_unseen.document_info.author == ""
    # publication date is not set for unseen papers
    assert inference_data_unseen.document_info.publication_date == ""
    # no arxiv labels for unseen papers
    assert inference_data_unseen.document_info.arxiv_labels == []
    assert (
        inference_data_unseen.document_info.semanticscholar_url
        == "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    )
    assert inference_data_unseen.document_info.arxiv_url == "https://arxiv.org/abs/2303.08774"
    assert len(inference_data_unseen.document_info.abstract) > 0


def test_document_info_defaults() -> None:
    document_info = DocumentInfo(d3_document_id=3)

    assert document_info.d3_document_id == 3
    assert document_info.title == ""
    assert document_info.author == ""
    assert document_info.publication_date == ""
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
            "2000-01-01",
            ["cs.AI", "cs.CL"],
            "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            "https://arxiv.org/abs/2106.01572",
            "Abstract",
        )
