import dataclasses

import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.inference.features import Ranks

feature_fixtures_skip_ci = [
    lazy_fixture("inference_data_seen_ranks"),
    lazy_fixture("inference_data_constructor_seen_ranks"),
]

feature_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_unseen_ranks"),
    lazy_fixture("inference_data_constructor_unseen_ranks"),
]


@pytest.mark.parametrize(
    "ranks",
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
def test_rank_attributes(ranks: Ranks) -> None:
    assert isinstance(ranks, Ranks)
    assert list(dataclasses.asdict(ranks)) == [
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis",
        "bibliographic_coupling",
    ]


@pytest.mark.parametrize(
    "ranks",
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
def test_publication_date_rank(ranks: Ranks) -> None:
    assert isinstance(ranks.publication_date, pl.DataFrame)

    assert ranks.publication_date.width == 2
    assert ranks.publication_date.columns == [
        "candidate_d3_document_id",
        "publication_date_rank",
    ]

    assert ranks.publication_date["candidate_d3_document_id"].dtype == pl.Int64
    assert ranks.publication_date["publication_date_rank"].dtype == pl.Float32

    # check that best rank of 1 is assigned
    assert ranks.publication_date["publication_date_rank"].min() == 1
    assert ranks.publication_date["publication_date_rank"].is_between(1, 101).all()


@pytest.mark.parametrize(
    "ranks",
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
def test_citationcount_document_rank(ranks: Ranks) -> None:
    assert isinstance(ranks.citationcount_document, pl.DataFrame)

    assert ranks.citationcount_document["candidate_d3_document_id"].dtype == pl.Int64
    assert ranks.citationcount_document["citationcount_document_rank"].dtype == pl.Float32

    assert ranks.citationcount_document.width == 2
    assert ranks.citationcount_document.columns == [
        "candidate_d3_document_id",
        "citationcount_document_rank",
    ]

    # check that best rank of 1 is assigned
    assert ranks.citationcount_document["citationcount_document_rank"].min() == 1
    assert ranks.citationcount_document["citationcount_document_rank"].is_between(1, 101).all()


@pytest.mark.parametrize(
    "ranks",
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
def test_citationcount_author_rank(ranks: Ranks) -> None:
    assert isinstance(ranks.citationcount_author, pl.DataFrame)

    assert ranks.citationcount_author["candidate_d3_document_id"].dtype == pl.Int64
    assert ranks.citationcount_author["citationcount_author_rank"].dtype == pl.Float32

    assert ranks.citationcount_author.width == 2
    assert ranks.citationcount_author.columns == [
        "candidate_d3_document_id",
        "citationcount_author_rank",
    ]

    # check that best rank of 1 is assigned
    assert ranks.citationcount_author["citationcount_author_rank"].min() == 1
    assert ranks.citationcount_author["citationcount_author_rank"].is_between(1, 101).all()


@pytest.mark.parametrize(
    "ranks",
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
def test_co_citation_analysis_rank(ranks: Ranks) -> None:
    assert isinstance(ranks.co_citation_analysis, pl.DataFrame)

    assert ranks.co_citation_analysis["candidate_d3_document_id"].dtype == pl.Int64
    assert ranks.co_citation_analysis["co_citation_analysis_rank"].dtype == pl.Float32

    assert ranks.co_citation_analysis.width == 2
    assert ranks.co_citation_analysis.columns == [
        "candidate_d3_document_id",
        "co_citation_analysis_rank",
    ]

    # check that best rank of 1 is assigned
    assert ranks.co_citation_analysis["co_citation_analysis_rank"].min() == 1
    assert ranks.co_citation_analysis["co_citation_analysis_rank"].is_between(1, 101).all()


@pytest.mark.parametrize(
    "ranks",
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
def test_bibliographic_coupling_rank(ranks: Ranks) -> None:
    assert isinstance(ranks.bibliographic_coupling, pl.DataFrame)

    assert ranks.bibliographic_coupling["candidate_d3_document_id"].dtype == pl.Int64
    assert ranks.bibliographic_coupling["bibliographic_coupling_rank"].dtype == pl.Float32

    assert ranks.bibliographic_coupling.width == 2
    assert ranks.bibliographic_coupling.columns == [
        "candidate_d3_document_id",
        "bibliographic_coupling_rank",
    ]

    # check that best rank of 1 is assigned
    assert ranks.bibliographic_coupling["bibliographic_coupling_rank"].min() == 1
    assert ranks.bibliographic_coupling["bibliographic_coupling_rank"].is_between(1, 101).all()


@pytest.mark.parametrize(
    "ranks",
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
def test_no_missing_values_ranks(ranks: Ranks) -> None:
    assert ranks.publication_date.null_count().sum(axis=1).item() == 0
    assert ranks.citationcount_document.null_count().sum(axis=1).item() == 0
    assert ranks.citationcount_author.null_count().sum(axis=1).item() == 0
    assert ranks.co_citation_analysis.null_count().sum(axis=1).item() == 0
    assert ranks.bibliographic_coupling.null_count().sum(axis=1).item() == 0


def test_kw_only_initialization_ranks() -> None:
    with pytest.raises(TypeError):
        Ranks(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
