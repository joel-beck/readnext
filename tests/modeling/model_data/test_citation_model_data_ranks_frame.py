import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import CitationRanksFrame

citation_ranks_frame_fixtures = [
    lazy_fixture("citation_model_data_constructor_ranks_frame"),
    lazy_fixture("citation_model_data_ranks_frame"),
]

citation_ranks_frame_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_citation_model_data_ranks_frame"),
    lazy_fixture("inference_data_constructor_seen_citation_model_data_ranks_frame"),
]

citation_ranks_frame_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_citation_model_data_ranks_frame"),
    lazy_fixture("inference_data_constructor_unseen_citation_model_data_ranks_frame"),
]


@pytest.mark.parametrize(
    "citation_ranks_frame",
    [
        *[pytest.param(fixture) for fixture in citation_ranks_frame_fixtures],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in citation_ranks_frame_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in citation_ranks_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_citation_model_data_ranks_frame(citation_ranks_frame: CitationRanksFrame) -> None:
    assert isinstance(citation_ranks_frame, pl.DataFrame)

    assert citation_ranks_frame.width == 6
    assert citation_ranks_frame.columns == [
        "candidate_d3_document_id",
        "publication_date_rank",
        "citationcount_document_rank",
        "citationcount_author_rank",
        "co_citation_analysis_rank",
        "bibliographic_coupling_rank",
    ]

    assert citation_ranks_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert citation_ranks_frame["publication_date_rank"].dtype == pl.Float32
    assert citation_ranks_frame["citationcount_document_rank"].dtype == pl.Float32
    assert citation_ranks_frame["citationcount_author_rank"].dtype == pl.Float32
    assert citation_ranks_frame["co_citation_analysis_rank"].dtype == pl.Float32
    assert citation_ranks_frame["bibliographic_coupling_rank"].dtype == pl.Float32

    # check that all ranks are between 1 and 101 where at least one rank is exactly 1
    rank_columns = citation_ranks_frame.select(pl.exclude("candidate_d3_document_id"))

    column_minimums = rank_columns.min()
    assert (column_minimums.min(axis=1) == 1.0).all()
    assert (column_minimums.max(axis=1) == 1.0).all()

    column_maximums = rank_columns.max()
    assert (column_maximums.max(axis=1) <= 101.0).all()
    assert (column_maximums.min(axis=1) <= 101.0).all()
