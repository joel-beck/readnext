import dataclasses

import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.inference import DocumentIdentifier

document_identifier_fixtures = [
    lazy_fixture("inference_data_document_identifier"),
    lazy_fixture("inference_data_constructor_document_identifier"),
]


@pytest.mark.parametrize("document_identifier", document_identifier_fixtures)
def test_inference_data_document_identifier(
    document_identifier: DocumentIdentifier,
) -> None:
    assert isinstance(document_identifier, DocumentIdentifier)
    assert list(dataclasses.asdict(document_identifier)) == [
        "d3_document_id",
        "semanticscholar_id",
        "semanticscholar_url",
        "arxiv_id",
        "arxiv_url",
    ]


def test_inference_data_document_identifier_seen(
    inference_data_document_identifier_seen: DocumentIdentifier,
) -> None:
    assert inference_data_document_identifier_seen.d3_document_id == 13756489
    assert (
        inference_data_document_identifier_seen.semanticscholar_id
        == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    assert (
        inference_data_document_identifier_seen.semanticscholar_url
        == "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    assert inference_data_document_identifier_seen.arxiv_id == "1706.03762"
    assert inference_data_document_identifier_seen.arxiv_url == "https://arxiv.org/abs/1706.03762"


def test_inference_data_document_identifier_unseen(
    inference_data_document_identifier_unseen: DocumentIdentifier,
) -> None:
    assert inference_data_document_identifier_unseen.d3_document_id == -1
    assert (
        inference_data_document_identifier_unseen.semanticscholar_id
        == "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    )
    assert (
        inference_data_document_identifier_unseen.semanticscholar_url
        == "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    )
    assert inference_data_document_identifier_unseen.arxiv_id == "2303.08774"
    assert inference_data_document_identifier_unseen.arxiv_url == "https://arxiv.org/abs/2303.08774"


def test_kw_only_initialization_document_identifier() -> None:
    with pytest.raises(TypeError):
        DocumentIdentifier(
            -1,  # type: ignore
            "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "2303.08774",
            "https://arxiv.org/abs/2303.08774",
        )
