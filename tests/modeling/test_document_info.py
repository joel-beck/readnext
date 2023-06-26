import dataclasses

import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import DocumentInfo

document_info_fixtures_seen = [
    lazy_fixture("model_data_constructor_plugin_seen_query_document"),
    lazy_fixture("model_data_seen_query_document"),
]

document_info_fixtures_unseen = [
    lazy_fixture("model_data_constructor_plugin_unseen_query_document"),
    lazy_fixture("model_data_unseen_query_document"),
]

document_info_fixtures = document_info_fixtures_seen + document_info_fixtures_unseen

document_info_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_model_data_query_document"),
    lazy_fixture("inference_data_constructor_seen_document_info"),
    lazy_fixture("inference_data_seen_document_info"),
]

document_info_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_model_data_query_document"),
    lazy_fixture("inference_data_constructor_unseen_document_info"),
    lazy_fixture("inference_data_unseen_document_info"),
]


@pytest.mark.updated
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


@pytest.mark.updated
@pytest.mark.parametrize(
    "document_info",
    [
        *[pytest.param(fixture) for fixture in document_info_fixtures],
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in document_info_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in document_info_fixtures_slow_skip_ci
        ],
    ],
)
def test_from_data(document_info: DocumentInfo) -> None:
    assert isinstance(document_info, DocumentInfo)
    assert list(dataclasses.asdict(document_info)) == [
        "d3_document_id",
        "title",
        "author",
        "publication_date",
        "arxiv_labels",
        "semanticscholar_url",
        "arxiv_url",
        "abstract",
    ]


@pytest.mark.updated
@pytest.mark.parametrize(
    "document_info",
    [
        *[pytest.param(fixture) for fixture in document_info_fixtures_seen],
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in document_info_fixtures_skip_ci
        ],
    ],
)
def test_from_data_seen(document_info: DocumentInfo) -> None:
    assert document_info.d3_document_id == 13756489
    assert document_info.title == "Attention is All you Need"
    assert document_info.author == "Lukasz Kaiser"
    assert document_info.publication_date == "2017-06-12"
    assert document_info.arxiv_labels == ["cs.CL", "cs.LG"]
    assert (
        document_info.semanticscholar_url
        == "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    assert document_info.arxiv_url == "https://arxiv.org/abs/1706.03762"
    assert len(document_info.abstract) > 0


@pytest.mark.updated
@pytest.mark.parametrize(
    "document_info",
    [
        pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
        for fixture in document_info_fixtures_slow_skip_ci
    ],
)
def test_from_inference_data_unseen(document_info: DocumentInfo) -> None:
    assert document_info.d3_document_id == -1
    assert document_info.title == "GPT-4 Technical Report"
    # author is not set for unseen papers
    assert document_info.author == ""
    # publication date is not set for unseen papers
    assert document_info.publication_date == ""
    # no arxiv labels for unseen papers
    assert document_info.arxiv_labels == []
    # semanticscholar url is not set since input identifier is arxiv url
    assert (
        document_info.semanticscholar_url
        == "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    )
    assert document_info.arxiv_url == ""
    assert len(document_info.abstract) > 0


@pytest.mark.updated
@pytest.mark.parametrize("document_info", document_info_fixtures_unseen)
def test_from_model_data_unseen(
    document_info: DocumentInfo,
) -> None:
    assert document_info.d3_document_id == -1
    assert document_info.title == "TestTitle"
    assert document_info.abstract == "TestAbstract"
    assert document_info.author == ""
    assert document_info.publication_date == ""
    assert document_info.arxiv_labels == []
    assert document_info.semanticscholar_url == "TestURL"
    assert document_info.arxiv_url == "ArxivURL"


@pytest.mark.updated
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
        "Publication Date: \n"
        "Arxiv Labels: []\n"
        "Semanticscholar URL: \n"
        "Arxiv URL: "
    )
    assert str(document_info) == str_representation


@pytest.mark.updated
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
