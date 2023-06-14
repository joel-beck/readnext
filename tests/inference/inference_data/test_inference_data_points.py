import dataclasses

import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.inference.features import Points

points_fixtures = [
    lazy_fixture("inference_data_points"),
    lazy_fixture("inference_data_constructor_points"),
]


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("points", points_fixtures)
def test_points_attributes(points: Points) -> None:
    assert isinstance(points, Points)
    assert list(dataclasses.asdict(points)) == [
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis",
        "bibliographic_coupling",
    ]


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("points", points_fixtures)
def test_publication_date_points(points: Points) -> None:
    assert isinstance(points.publication_date, pl.DataFrame)

    assert points.publication_date.shape[1] == 2
    assert points.publication_date.columns == [
        "candidate_d3_document_id",
        "publication_date_points",
    ]

    assert points.publication_date["candidate_d3_document_id"].dtype == pl.Int64
    assert points.publication_date["publication_date_points"].dtype == pl.Float32

    # check that min and max rank are correct
    assert points.publication_date["publication_date_points"].min() == 0
    assert points.publication_date["publication_date_points"].max() == 100
    assert points.publication_date["publication_date_points"].is_between(0, 100).all()


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("points", points_fixtures)
def test_citationcount_document_points(points: Points) -> None:
    assert isinstance(points.citationcount_document, pl.DataFrame)

    assert points.citationcount_document["candidate_d3_document_id"].dtype == pl.Int64
    assert points.citationcount_document["citationcount_document_points"].dtype == pl.Float32

    assert points.citationcount_document.shape[1] == 2
    assert points.citationcount_document.columns == [
        "candidate_d3_document_id",
        "citationcount_document_points",
    ]

    # check that min and max rank are correct
    assert points.citationcount_document["citationcount_document_points"].min() == 0
    assert points.citationcount_document["citationcount_document_points"].max() == 100
    assert points.citationcount_document["citationcount_document_points"].is_between(0, 100).all()


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("points", points_fixtures)
def test_citationcount_author_points(points: Points) -> None:
    assert isinstance(points.citationcount_author, pl.DataFrame)

    assert points.citationcount_author["candidate_d3_document_id"].dtype == pl.Int64
    assert points.citationcount_author["citationcount_author_points"].dtype == pl.Float32

    assert points.citationcount_author.shape[1] == 2
    assert points.citationcount_author.columns == [
        "candidate_d3_document_id",
        "citationcount_author_points",
    ]

    # check that min and max rank are correct
    assert points.citationcount_author["citationcount_author_points"].min() == 0
    assert points.citationcount_author["citationcount_author_points"].max() == 100
    assert points.citationcount_author["citationcount_author_points"].is_between(0, 100).all()


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("points", points_fixtures)
def test_co_citation_analysis_points(points: Points) -> None:
    assert isinstance(points.co_citation_analysis, pl.DataFrame)

    assert points.co_citation_analysis["candidate_d3_document_id"].dtype == pl.Int64
    assert points.co_citation_analysis["co_citation_analysis_points"].dtype == pl.Float32

    assert points.co_citation_analysis.shape[1] == 2
    assert points.co_citation_analysis.columns == [
        "candidate_d3_document_id",
        "co_citation_analysis_points",
    ]

    # check that min and max rank are correct
    assert points.co_citation_analysis["co_citation_analysis_points"].min() == 0
    assert points.co_citation_analysis["co_citation_analysis_points"].max() == 100
    assert points.co_citation_analysis["co_citation_analysis_points"].is_between(0, 100).all()


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("points", points_fixtures)
def test_bibliographic_coupling_points(points: Points) -> None:
    assert isinstance(points.bibliographic_coupling, pl.DataFrame)

    assert points.bibliographic_coupling["candidate_d3_document_id"].dtype == pl.Int64
    assert points.bibliographic_coupling["bibliographic_coupling_points"].dtype == pl.Float32

    assert points.bibliographic_coupling.shape[1] == 2
    assert points.bibliographic_coupling.columns == [
        "candidate_d3_document_id",
        "bibliographic_coupling_points",
    ]

    # check that min and max rank are correct
    assert points.bibliographic_coupling["bibliographic_coupling_points"].min() == 0
    assert points.bibliographic_coupling["bibliographic_coupling_points"].max() == 100
    assert points.bibliographic_coupling["bibliographic_coupling_points"].is_between(0, 100).all()


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("points", points_fixtures)
def test_no_missing_values_points(points: Points) -> None:
    assert points.publication_date.null_count().sum(axis=1).item() == 0
    assert points.citationcount_document.null_count().sum(axis=1).item() == 0
    assert points.citationcount_author.null_count().sum(axis=1).item() == 0
    assert points.co_citation_analysis.null_count().sum(axis=1).item() == 0
    assert points.bibliographic_coupling.null_count().sum(axis=1).item() == 0


@pytest.mark.updated
def test_kw_only_initialization_points() -> None:
    with pytest.raises(TypeError):
        Points(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
