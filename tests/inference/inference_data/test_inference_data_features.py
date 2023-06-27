import dataclasses
import re

import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference.features import Features

feature_fixtures_skip_ci = [
    lazy_fixture("inference_data_seen_features"),
    lazy_fixture("inference_data_constructor_seen_features"),
]

feature_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_unseen_features"),
    lazy_fixture("inference_data_constructor_unseen_features"),
]


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
    ],
)
def test_feature_attributes(features: Features) -> None:
    assert isinstance(features, Features)
    assert list(dataclasses.asdict(features)) == [
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis",
        "bibliographic_coupling",
        "cosine_similarity",
        "feature_weights",
    ]


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
    ],
)
def test_publication_date(features: Features) -> None:
    assert isinstance(features.publication_date, pl.DataFrame)

    assert features.publication_date.shape[1] == 2
    assert features.publication_date.columns == [
        "candidate_d3_document_id",
        "publication_date",
    ]

    assert features.publication_date["candidate_d3_document_id"].dtype == pl.Int64
    assert features.publication_date["publication_date"].dtype == pl.Utf8

    # check that all dates have the format YYYY-MM-DD
    assert all(
        re.match(r"^\d{4}-\d{2}-\d{2}$", date)
        for date in features.publication_date["publication_date"]
    )


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
    ],
)
def test_citationcount_document(features: Features) -> None:
    assert isinstance(features.citationcount_document, pl.DataFrame)

    assert features.citationcount_document.shape[1] == 2
    assert features.citationcount_document.columns == [
        "candidate_d3_document_id",
        "citationcount_document",
    ]

    assert features.citationcount_document["candidate_d3_document_id"].dtype == pl.Int64
    assert features.citationcount_document["citationcount_document"].dtype == pl.Int64

    # check that all citation counts are non-negative
    assert all(
        citation_count >= 0
        for citation_count in features.citationcount_document["citationcount_document"]
    )


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
    ],
)
def test_citationcount_author(features: Features) -> None:
    assert isinstance(features.citationcount_author, pl.DataFrame)

    assert features.citationcount_author.shape[1] == 2
    assert features.citationcount_author.columns == [
        "candidate_d3_document_id",
        "citationcount_author",
    ]

    assert features.citationcount_author["candidate_d3_document_id"].dtype == pl.Int64
    assert features.citationcount_author["citationcount_author"].dtype == pl.Int64

    # check that all citation counts are non-negative
    assert all(
        citation_count >= 0
        for citation_count in features.citationcount_author["citationcount_author"]
    )


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
    ],
)
def test_co_citation_analysis(features: Features) -> None:
    assert isinstance(features.co_citation_analysis, pl.DataFrame)

    assert features.co_citation_analysis.shape[1] == 2
    assert features.co_citation_analysis.columns == [
        "candidate_d3_document_id",
        "co_citation_analysis_score",
    ]

    assert features.co_citation_analysis["candidate_d3_document_id"].dtype == pl.Int64
    assert features.co_citation_analysis["co_citation_analysis_score"].dtype == pl.Int64

    # check that all scores are non-negative
    assert all(score >= 0 for score in features.co_citation_analysis["co_citation_analysis_score"])


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
    ],
)
def test_bibliographic_coupling(features: Features) -> None:
    assert isinstance(features.bibliographic_coupling, pl.DataFrame)

    assert features.bibliographic_coupling.shape[1] == 2
    assert features.bibliographic_coupling.columns == [
        "candidate_d3_document_id",
        "bibliographic_coupling_score",
    ]

    assert features.bibliographic_coupling["candidate_d3_document_id"].dtype == pl.Int64
    assert features.bibliographic_coupling["bibliographic_coupling_score"].dtype == pl.Int64

    # check that all scores are non-negative
    assert all(
        score >= 0 for score in features.bibliographic_coupling["bibliographic_coupling_score"]
    )


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
    ],
)
def test_cosine_similarity(features: Features) -> None:
    assert isinstance(features.cosine_similarity, pl.DataFrame)

    assert features.cosine_similarity.shape[1] == 2
    assert features.cosine_similarity.columns == [
        "candidate_d3_document_id",
        "cosine_similarity",
    ]

    assert features.cosine_similarity["candidate_d3_document_id"].dtype == pl.Int64
    assert features.cosine_similarity["cosine_similarity"].dtype == pl.Float64

    # check that all scores are between -1 and 1
    assert all(
        features.cosine_similarity["cosine_similarity"].is_between(lower_bound=-1, upper_bound=1)
    )


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
    ],
)
def test_no_missing_values(features: Features) -> None:
    assert features.publication_date.null_count().sum(axis=1).item() == 0
    assert features.citationcount_document.null_count().sum(axis=1).item() == 0
    assert features.citationcount_author.null_count().sum(axis=1).item() == 0
    assert features.co_citation_analysis.null_count().sum(axis=1).item() == 0
    assert features.bibliographic_coupling.null_count().sum(axis=1).item() == 0
    assert features.cosine_similarity.null_count().sum(axis=1).item() == 0


def test_kw_only_initialization_features() -> None:
    with pytest.raises(TypeError):
        Features(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            FeatureWeights(),
        )
