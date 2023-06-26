import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import LanguageModelData
from readnext.modeling.document_info import DocumentInfo

language_model_data_fixtures = [
    lazy_fixture("language_model_data_seen"),
    lazy_fixture("language_model_data_unseen"),
]

language_model_data_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_language_model_data"),
    lazy_fixture("inference_data_constructor_seen_language_model_data"),
]

language_model_data_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_language_model_data"),
    lazy_fixture("inference_data_constructor_unseen_language_model_data"),
]


@pytest.mark.parametrize(
    "language_model_data",
    [
        *[pytest.param(fixture) for fixture in language_model_data_fixtures],
        *[
            pytest.param(fixture, marks=(pytest.mark.skip_ci))
            for fixture in language_model_data_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in language_model_data_fixtures_slow_skip_ci
        ],
    ],
)
def test_language_model_data_getitem(language_model_data: LanguageModelData) -> None:
    d3_document_ids = [206594692, 6628106, 14124313]
    sliced_language_model_data = language_model_data[d3_document_ids]

    # query_document unchanged through slicing
    assert language_model_data.query_document == sliced_language_model_data.query_document

    assert (
        sliced_language_model_data.integer_labels_frame["d3_document_id"].to_list()
        == d3_document_ids
    )
    assert sliced_language_model_data.info_frame["d3_document_id"].to_list() == d3_document_ids
    assert sliced_language_model_data.features_frame["d3_document_id"].to_list() == d3_document_ids


@pytest.mark.updated
def test_kw_only_initialization_language_model_data() -> None:
    with pytest.raises(TypeError):
        LanguageModelData(
            DocumentInfo(d3_document_id=1),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
