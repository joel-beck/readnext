import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import DocumentInfo

# SUBSECTION: Test Query Document
query_document_fixtures_seen = [
    "citation_model_data_query_document",
    "language_model_data_query_document",
    "citation_model_data_constructor_query_document",
    "language_model_data_constructor_query_document",
]

query_document_fixtures_seen_slow_skip_ci = [
    "seen_paper_attribute_getter_citation_model_data_query_document",
    "seen_paper_attribute_getter_language_model_data_query_document",
    "inference_data_constructor_seen_citation_model_data_query_document",
    "inference_data_constructor_seen_language_model_data_query_document",
]


@pytest.mark.parametrize(
    "query_document",
    [
        *[lazy_fixture(fixture) for fixture in query_document_fixtures_seen],
        *[
            pytest.param(
                lazy_fixture(fixture),
                marks=(pytest.mark.slow, pytest.mark.skip_ci),
            )
            for fixture in query_document_fixtures_seen_slow_skip_ci
        ],
    ],
)
def test_seen_model_data_query_document(query_document: DocumentInfo) -> None:
    assert isinstance(query_document, DocumentInfo)

    assert isinstance(query_document.d3_document_id, int)
    assert query_document.d3_document_id == 13756489

    assert isinstance(query_document.title, str)
    assert query_document.title == "Attention is All you Need"

    assert isinstance(query_document.author, str)
    assert query_document.author == "Lukasz Kaiser"

    assert isinstance(query_document.arxiv_labels, list)
    assert all(isinstance(label, str) for label in query_document.arxiv_labels)
    assert query_document.arxiv_labels == ["cs.CL", "cs.LG"]

    assert isinstance(query_document.abstract, str)
    # abstract is not set for seen papers
    assert query_document.abstract == ""


query_document_fixtures_unseen_skip_ci = [
    "unseen_paper_attribute_getter_citation_model_data_query_document",
    "unseen_paper_attribute_getter_language_model_data_query_document",
    "inference_data_constructor_unseen_citation_model_data_query_document",
    "inference_data_constructor_unseen_language_model_data_query_document",
]


@pytest.mark.parametrize(
    "query_document",
    [
        pytest.param(
            lazy_fixture(fixture),
            marks=(pytest.mark.skip_ci),
        )
        for fixture in query_document_fixtures_unseen_skip_ci
    ],
)
def test_unseen_model_data_query_document(query_document: DocumentInfo) -> None:
    assert isinstance(query_document, DocumentInfo)

    assert isinstance(query_document.d3_document_id, int)
    assert query_document.d3_document_id == -1

    assert isinstance(query_document.title, str)
    assert query_document.title == "GPT-4 Technical Report"

    assert isinstance(query_document.author, str)
    # author is not set for unseen papers
    assert query_document.author == ""

    assert isinstance(query_document.arxiv_labels, list)
    assert all(isinstance(label, str) for label in query_document.arxiv_labels)
    # no arxiv labels for unseen papers
    assert query_document.arxiv_labels == []

    assert isinstance(query_document.abstract, str)
    # abstract is set for unseen papers
    assert len(query_document.abstract) > 0


# SUBSECTION: Test Integer Labels
integer_labels_fixtures_seen = [
    "citation_model_data_integer_labels",
    "language_model_data_integer_labels",
    "citation_model_data_constructor_integer_labels",
    "language_model_data_constructor_integer_labels",
]

integer_labels_fixtures_seen_slow_skip_ci = [
    "seen_paper_attribute_getter_citation_model_data_integer_labels",
    "seen_paper_attribute_getter_language_model_data_integer_labels",
    "inference_data_constructor_seen_citation_model_data_integer_labels",
    "inference_data_constructor_seen_language_model_data_integer_labels",
]

integer_labels_fixtures_unseen_skip_ci = [
    "unseen_paper_attribute_getter_citation_model_data_integer_labels",
    "unseen_paper_attribute_getter_language_model_data_integer_labels",
    "inference_data_constructor_unseen_citation_model_data_integer_labels",
    "inference_data_constructor_unseen_language_model_data_integer_labels",
]


@pytest.mark.parametrize(
    "integer_labels",
    [
        *[lazy_fixture(fixture) for fixture in integer_labels_fixtures_seen],
        *[
            pytest.param(
                lazy_fixture(fixture),
                marks=(pytest.mark.slow, pytest.mark.skip_ci),
            )
            for fixture in integer_labels_fixtures_seen_slow_skip_ci
        ],
        *[
            pytest.param(
                lazy_fixture(fixture),
                marks=(pytest.mark.skip_ci),
            )
            for fixture in integer_labels_fixtures_unseen_skip_ci
        ],
    ],
)
def test_model_data_integer_labels(integer_labels: pd.Series) -> None:
    assert isinstance(integer_labels, pd.Series)

    assert integer_labels.dtype == np.int64  # type: ignore
    assert integer_labels.name == "integer_labels"
    assert integer_labels.index.name == "document_id"


@pytest.mark.parametrize(
    "integer_labels",
    [
        *[lazy_fixture(fixture) for fixture in integer_labels_fixtures_seen],
        *[
            pytest.param(
                lazy_fixture(fixture),
                marks=(pytest.mark.slow, pytest.mark.skip_ci),
            )
            for fixture in integer_labels_fixtures_seen_slow_skip_ci
        ],
    ],
)
def test_seen_model_data_integer_labels(integer_labels: pd.Series) -> None:
    assert integer_labels.unique().tolist() == [0, 1]


@pytest.mark.parametrize(
    "integer_labels",
    [
        pytest.param(
            lazy_fixture(fixture),
            marks=(pytest.mark.slow, pytest.mark.skip_ci),
        )
        for fixture in integer_labels_fixtures_unseen_skip_ci
    ],
)
def test_unseen_model_data_integer_labels(integer_labels: pd.Series) -> None:
    assert integer_labels.unique().tolist() == [0]
