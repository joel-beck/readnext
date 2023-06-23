import dataclasses

import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.inference import DocumentIdentifier


constructor_plugins_seen_document_identifiers = [
    lazy_fixture(
        "inference_data_constructor_plugin_seen_document_identifier_from_semanticscholar_id"
    ),
    lazy_fixture(
        "inference_data_constructor_plugin_seen_document_identifier_from_semanticscholar_id"
    ),
    lazy_fixture("inference_data_constructor_plugin_seen_document_identifier_from_arxiv_id"),
    lazy_fixture("inference_data_constructor_plugin_seen_document_identifier_from_arxiv_url"),
]

constructor_plugins_unseen_document_identifiers = [
    lazy_fixture(
        "inference_data_constructor_plugin_unseen_document_identifier_from_semanticscholar_id"
    ),
    lazy_fixture(
        "inference_data_constructor_plugin_unseen_document_identifier_from_semanticscholar_id"
    ),
    lazy_fixture("inference_data_constructor_plugin_unseen_document_identifier_from_arxiv_id"),
    lazy_fixture("inference_data_constructor_plugin_unseen_document_identifier_from_arxiv_url"),
]

document_identifier_fixtures_skip_ci = [
    lazy_fixture("inference_data_seen_document_identifier"),
    lazy_fixture("inference_data_constructor_seen_document_identifier"),
]

document_identifier_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_unseen_document_identifier"),
    lazy_fixture("inference_data_constructor_unseen_document_identifier"),
]


@pytest.mark.updated
@pytest.mark.parametrize(
    "document_identifier",
    [
        *constructor_plugins_seen_document_identifiers,
        *constructor_plugins_unseen_document_identifiers,
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in document_identifier_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in document_identifier_fixtures_slow_skip_ci
        ],
    ],
)
def test_inference_data_document_identifier(document_identifier: DocumentIdentifier) -> None:
    assert isinstance(document_identifier, DocumentIdentifier)
    assert list(dataclasses.asdict(document_identifier)) == [
        "d3_document_id",
        "semanticscholar_id",
        "semanticscholar_url",
        "arxiv_id",
        "arxiv_url",
    ]


@pytest.mark.updated
@pytest.mark.parametrize(
    "document_identifier",
    [
        *constructor_plugins_seen_document_identifiers,
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in document_identifier_fixtures_skip_ci
        ],
    ],
)
def test_inference_data_seen_document_identifier(
    document_identifier: DocumentIdentifier,
) -> None:
    assert document_identifier.d3_document_id == 13756489
    assert document_identifier.semanticscholar_id == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    assert (
        document_identifier.semanticscholar_url
        == "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    assert document_identifier.arxiv_id == "1706.03762"
    assert document_identifier.arxiv_url == "https://arxiv.org/abs/1706.03762"


@pytest.mark.updated
@pytest.mark.parametrize(
    "document_identifier",
    [
        *constructor_plugins_unseen_document_identifiers,
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in document_identifier_fixtures_slow_skip_ci
        ],
    ],
)
def test_inference_data_unseen_document_identifier(
    document_identifier: DocumentIdentifier,
) -> None:
    assert document_identifier.d3_document_id == -1
    assert document_identifier.semanticscholar_id == "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    assert (
        document_identifier.semanticscholar_url
        == "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    )
    assert document_identifier.arxiv_id == "2303.08774"
    assert document_identifier.arxiv_url == "https://arxiv.org/abs/2303.08774"


@pytest.mark.updated
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
