import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import CitationPointsFrame

citation_points_frame_fixtures = [
    lazy_fixture("citation_model_data_constructor_points_frame"),
    lazy_fixture("citation_model_data_points_frame"),
]

citation_points_frame_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_citation_model_data_points_frame"),
    lazy_fixture("inference_data_constructor_seen_citation_model_data_points_frame"),
]

citation_points_frame_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_citation_model_data_points_frame"),
    lazy_fixture("inference_data_constructor_unseen_citation_model_data_points_frame"),
]


@pytest.mark.parametrize(
    "citation_points_frame",
    [
        *[pytest.param(fixture) for fixture in citation_points_frame_fixtures],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in citation_points_frame_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in citation_points_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_citation_model_data_points_frame(citation_points_frame: CitationPointsFrame) -> None:
    assert isinstance(citation_points_frame, pl.DataFrame)

    assert citation_points_frame.width == 6
    assert citation_points_frame.columns == [
        "candidate_d3_document_id",
        "publication_date_points",
        "citationcount_document_points",
        "citationcount_author_points",
        "co_citation_analysis_points",
        "bibliographic_coupling_points",
    ]

    assert citation_points_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert citation_points_frame["publication_date_points"].dtype == pl.Float32
    assert citation_points_frame["citationcount_document_points"].dtype == pl.Float32
    assert citation_points_frame["citationcount_author_points"].dtype == pl.Float32
    assert citation_points_frame["co_citation_analysis_points"].dtype == pl.Float32
    assert citation_points_frame["bibliographic_coupling_points"].dtype == pl.Float32

    # check that all points are between 0 and 100 where at least one score is exactly
    # 100
    rank_columns = citation_points_frame.select(pl.exclude("candidate_d3_document_id"))

    column_minimums = rank_columns.min()
    assert (column_minimums.min(axis=1) >= 0.0).all()
    assert (column_minimums.max(axis=1) >= 0.0).all()

    column_maximums = rank_columns.max()
    assert (column_maximums.max(axis=1) == 100.0).all()
    assert (column_maximums.min(axis=1) == 100.0).all()
