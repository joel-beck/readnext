import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import CitationFeaturesFrame, LanguageFeaturesFrame

citation_features_frame_fixtures = [
    lazy_fixture("citation_model_data_constructor_features_frame"),
    lazy_fixture("citation_model_data_features_frame"),
]

language_features_frame_fixtures = [
    lazy_fixture("language_model_data_constructor_features_frame"),
    lazy_fixture("language_model_data_features_frame"),
]

citation_features_frame_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_citation_model_data_features_frame"),
    lazy_fixture("inference_data_constructor_seen_citation_model_data_features_frame"),
]

language_features_frame_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_language_model_data_features_frame"),
    lazy_fixture("inference_data_constructor_seen_language_model_data_features_frame"),
]

citation_features_frame_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_citation_model_data_features_frame"),
    lazy_fixture("inference_data_constructor_unseen_citation_model_data_features_frame"),
]

language_features_frame_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_language_model_data_features_frame"),
    lazy_fixture("inference_data_constructor_unseen_language_model_data_features_frame"),
]


@pytest.mark.updated
@pytest.mark.parametrize(
    "citation_features_frame",
    [
        *[pytest.param(fixture) for fixture in citation_features_frame_fixtures],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in citation_features_frame_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in citation_features_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_citation_model_data_features_frame(citation_features_frame: CitationFeaturesFrame) -> None:
    assert isinstance(citation_features_frame, pl.DataFrame)

    assert citation_features_frame.shape[1] == 6
    assert citation_features_frame.columns == [
        "candidate_d3_document_id",
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis_score",
        "bibliographic_coupling_score",
    ]

    assert citation_features_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert citation_features_frame["publication_date"].dtype == pl.Utf8
    assert citation_features_frame["citationcount_document"].dtype == pl.Int64
    assert citation_features_frame["citationcount_author"].dtype == pl.Int64
    assert citation_features_frame["co_citation_analysis_score"].dtype == pl.Float64
    assert citation_features_frame["bibliographic_coupling_score"].dtype == pl.Float64


@pytest.mark.updated
@pytest.mark.parametrize(
    "language_features_frame",
    [
        *[pytest.param(fixture) for fixture in language_features_frame_fixtures],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in language_features_frame_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in language_features_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_language_model_data_features_frame(language_features_frame: LanguageFeaturesFrame) -> None:
    assert isinstance(language_features_frame, pl.DataFrame)

    assert language_features_frame.shape[1] == 2
    assert language_features_frame.columns == ["candidate_d3_document_id", "cosine_similarity"]

    assert language_features_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert language_features_frame["cosine_similarity"].dtype == pl.Float64
