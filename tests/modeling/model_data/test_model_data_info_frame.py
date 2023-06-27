import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import CitationInfoFrame, LanguageInfoFrame

citation_info_frame_fixtures = [
    lazy_fixture("citation_model_data_constructor_info_frame"),
    lazy_fixture("citation_model_data_info_frame"),
]

language_info_frame_fixtures = [
    lazy_fixture("language_model_data_constructor_info_frame"),
    lazy_fixture("language_model_data_info_frame"),
]

citation_info_frame_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_citation_model_data_info_frame"),
    lazy_fixture("inference_data_constructor_seen_citation_model_data_info_frame"),
]

language_info_frame_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_language_model_data_info_frame"),
    lazy_fixture("inference_data_constructor_seen_language_model_data_info_frame"),
]

citation_info_frame_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_citation_model_data_info_frame"),
    lazy_fixture("inference_data_constructor_unseen_citation_model_data_info_frame"),
]

language_info_frame_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_language_model_data_info_frame"),
    lazy_fixture("inference_data_constructor_unseen_language_model_data_info_frame"),
]


@pytest.mark.parametrize(
    "citation_info_frame",
    [
        *[pytest.param(fixture) for fixture in citation_info_frame_fixtures],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in citation_info_frame_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in citation_info_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_citation_model_data_info_frame(
    citation_info_frame: CitationInfoFrame,
) -> None:
    assert isinstance(citation_info_frame, pl.DataFrame)

    assert citation_info_frame.shape[1] == 6
    assert citation_info_frame.columns == [
        "candidate_d3_document_id",
        "title",
        "author",
        "arxiv_labels",
        "semanticscholar_url",
        "arxiv_url",
    ]

    assert citation_info_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert citation_info_frame["title"].dtype == pl.Utf8
    assert citation_info_frame["author"].dtype == pl.Utf8
    assert citation_info_frame["arxiv_labels"].dtype == pl.List(pl.Utf8)
    assert citation_info_frame["semanticscholar_url"].dtype == pl.Utf8
    assert citation_info_frame["arxiv_url"].dtype == pl.Utf8


@pytest.mark.parametrize(
    "language_info_frame",
    [
        *[pytest.param(fixture) for fixture in language_info_frame_fixtures],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in language_info_frame_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in language_info_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_language_model_data_info_frame(
    language_info_frame: LanguageInfoFrame,
) -> None:
    assert isinstance(language_info_frame, pl.DataFrame)

    assert language_info_frame.shape[1] == 7
    assert language_info_frame.columns == [
        "candidate_d3_document_id",
        "title",
        "author",
        "publication_date",
        "arxiv_labels",
        "semanticscholar_url",
        "arxiv_url",
    ]

    assert language_info_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert language_info_frame["title"].dtype == pl.Utf8
    assert language_info_frame["author"].dtype == pl.Utf8
    assert language_info_frame["arxiv_labels"].dtype == pl.List(pl.Utf8)
    assert language_info_frame["publication_date"].dtype == pl.Utf8
    assert language_info_frame["semanticscholar_url"].dtype == pl.Utf8
    assert language_info_frame["arxiv_url"].dtype == pl.Utf8
