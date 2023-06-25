import numpy as np
import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

# SUBSECTION: Test Query Document
# TODO: Add fixtures from inference data constructor plugin!
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


query_document_fixtures_unseen_skip_ci = [
    "unseen_paper_attribute_getter_citation_model_data_query_document",
    "unseen_paper_attribute_getter_language_model_data_query_document",
    "inference_data_constructor_unseen_citation_model_data_query_document",
    "inference_data_constructor_unseen_language_model_data_query_document",
]


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
def test_model_data_integer_labels(integer_labels: pl.Series) -> None:
    assert isinstance(integer_labels, pl.Series)

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
def test_seen_model_data_integer_labels(integer_labels: pl.Series) -> None:
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
def test_unseen_model_data_integer_labels(integer_labels: pl.Series) -> None:
    assert integer_labels.unique().tolist() == [0]
