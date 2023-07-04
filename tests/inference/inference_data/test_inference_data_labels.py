import dataclasses
import re

import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.inference.features import Labels

label_fixtures_skip_ci = [
    lazy_fixture("inference_data_seen_labels"),
    lazy_fixture("inference_data_constructor_seen_labels"),
]

label_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_unseen_labels"),
    lazy_fixture("inference_data_constructor_unseen_labels"),
]


@pytest.mark.parametrize(
    "labels",
    [
        *[pytest.param(fixture, marks=(pytest.mark.skip_ci)) for fixture in label_fixtures_skip_ci],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in label_fixtures_slow_skip_ci
        ],
    ],
)
def test_feature_attributes(labels: Labels) -> None:
    assert isinstance(labels, Labels)
    assert list(dataclasses.asdict(labels)) == ["arxiv", "integer"]


@pytest.mark.parametrize(
    "labels",
    [
        *[pytest.param(fixture, marks=(pytest.mark.skip_ci)) for fixture in label_fixtures_skip_ci],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in label_fixtures_slow_skip_ci
        ],
    ],
)
def test_arxiv_labels(labels: Labels) -> None:
    assert isinstance(labels.arxiv, pl.DataFrame)

    assert labels.arxiv.width == 2
    assert labels.arxiv.columns == [
        "candidate_d3_document_id",
        "arxiv_labels",
    ]

    assert labels.arxiv["candidate_d3_document_id"].dtype == pl.Int64
    assert labels.arxiv["arxiv_labels"].dtype == pl.List(pl.Utf8)

    # check that all arxiv labels contain at least one computer science label
    assert all(
        any(re.match(r"^cs\.", arxiv_label) for arxiv_label in arxiv_labels)
        for arxiv_labels in labels.arxiv["arxiv_labels"]
    )


@pytest.mark.parametrize(
    "labels",
    [
        *[pytest.param(fixture, marks=(pytest.mark.skip_ci)) for fixture in label_fixtures_skip_ci],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in label_fixtures_slow_skip_ci
        ],
    ],
)
def test_citationcount_document(labels: Labels) -> None:
    assert isinstance(labels.integer, pl.DataFrame)

    assert labels.integer.width == 2
    assert labels.integer.columns == [
        "candidate_d3_document_id",
        "integer_label",
    ]

    assert labels.integer["candidate_d3_document_id"].dtype == pl.Int64
    assert labels.integer["integer_label"].dtype == pl.Int64

    # check that all integer labels are either 0 or 1
    assert labels.integer["integer_label"].is_between(0, 1).all()


@pytest.mark.parametrize(
    "labels",
    [
        *[pytest.param(fixture, marks=(pytest.mark.skip_ci)) for fixture in label_fixtures_skip_ci],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in label_fixtures_slow_skip_ci
        ],
    ],
)
def test_no_missing_values(labels: Labels) -> None:
    assert labels.arxiv.null_count().sum(axis=1).item() == 0
    assert labels.integer.null_count().sum(axis=1).item() == 0


def test_kw_only_initialization_labels() -> None:
    with pytest.raises(TypeError):
        Labels(pl.DataFrame(), pl.DataFrame())  # type: ignore
