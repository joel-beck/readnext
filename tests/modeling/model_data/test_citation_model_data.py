import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import CitationModelData
from readnext.modeling.document_info import DocumentInfo

citation_model_data_fixtures = [
    lazy_fixture("citation_model_data_seen"),
    lazy_fixture("citation_model_data_unseen"),
]

citation_model_data_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_citation_model_data"),
    lazy_fixture("inference_data_constructor_seen_citation_model_data"),
]

citation_model_data_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_citation_model_data"),
    lazy_fixture("inference_data_constructor_unseen_citation_model_data"),
]


@pytest.mark.parametrize(
    "citation_model_data",
    [
        *[pytest.param(fixture) for fixture in citation_model_data_fixtures],
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in citation_model_data_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in citation_model_data_fixtures_slow_skip_ci
        ],
    ],
)
def test_citation_model_data_getitem(citation_model_data: CitationModelData) -> None:
    d3_document_ids = [206594692, 6628106, 14124313]
    sliced_citation_model_data = citation_model_data[d3_document_ids]

    # query_document unchanged through slicing
    assert citation_model_data.query_document == sliced_citation_model_data.query_document

    assert (
        sliced_citation_model_data.integer_labels_frame["candidate_d3_document_id"].to_list()
        == d3_document_ids
    )
    assert (
        sliced_citation_model_data.info_frame["candidate_d3_document_id"].to_list()
        == d3_document_ids
    )
    assert (
        sliced_citation_model_data.features_frame["candidate_d3_document_id"].to_list()
        == d3_document_ids
    )
    assert (
        sliced_citation_model_data.ranks_frame["candidate_d3_document_id"].to_list()
        == d3_document_ids
    )
    assert (
        sliced_citation_model_data.points_frame["candidate_d3_document_id"].to_list()
        == d3_document_ids
    )


def test_kw_only_initialization_citation_model_data() -> None:
    with pytest.raises(TypeError):
        CitationModelData(
            DocumentInfo(d3_document_id=1),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
