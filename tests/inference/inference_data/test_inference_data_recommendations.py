import polars as pl
import pytest
import dataclasses
from readnext.inference.features import Recommendations
from pytest_lazyfixture import lazy_fixture

feature_fixtures_skip_ci = [
    lazy_fixture("inference_data_seen_recommendations"),
    lazy_fixture("inference_data_constructor_seen_recommendations"),
]

feature_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_unseen_recommendations"),
    lazy_fixture("inference_data_constructor_unseen_recommendations"),
]


@pytest.mark.updated
@pytest.mark.parametrize(
    "recommendations",
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
def test_feature_attributes(recommendations: Recommendations) -> None:
    assert isinstance(recommendations, Recommendations)
    assert list(dataclasses.asdict(recommendations)) == [
        "citation_to_language_candidates",
        "citation_to_language",
        "language_to_citation_candidates",
        "language_to_citation",
    ]


@pytest.mark.updated
@pytest.mark.parametrize(
    "recommendations",
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
def test_citation_recommendations(recommendations: Recommendations) -> None:
    assert isinstance(recommendations.citation_to_language_candidates, pl.DataFrame)
    assert isinstance(recommendations.language_to_citation, pl.DataFrame)

    assert recommendations.citation_to_language_candidates.shape[1] == 18
    assert recommendations.language_to_citation.shape[1] == 18

    assert all(
        columns
        == [
            "candidate_d3_document_id",
            "weighted_points",
            "title",
            "author",
            "arxiv_labels",
            "integer_label",
            "semanticscholar_url",
            "arxiv_url",
            "publication_date",
            "publication_date_points",
            "citationcount_document",
            "citationcount_document_points",
            "citationcount_author",
            "citationcount_author_points",
            "co_citation_analysis_score",
            "co_citation_analysis_points",
            "bibliographic_coupling_score",
            "bibliographic_coupling_points",
        ]
        for columns in [
            recommendations.citation_to_language_candidates.columns,
            recommendations.language_to_citation.columns,
        ]
    )

    assert all(
        dtypes
        == [
            pl.Int64,
            pl.Float32,
            pl.Utf8,
            pl.Utf8,
            pl.List(pl.Utf8),
            pl.Int64,
            pl.Utf8,
            pl.Utf8,
            pl.Utf8,
            pl.Float32,
            pl.Int64,
            pl.Float32,
            pl.Int64,
            pl.Float32,
            pl.Int64,
            pl.Float32,
            pl.Int64,
            pl.Float32,
        ]
        for dtypes in [
            recommendations.citation_to_language_candidates.dtypes,
            recommendations.language_to_citation.dtypes,
        ]
    )

    # check that recommendations are sorted in descending order by weighted_points
    assert recommendations.citation_to_language_candidates["weighted_points"].is_sorted(
        descending=True
    )
    assert recommendations.language_to_citation["weighted_points"].is_sorted(descending=True)


@pytest.mark.updated
@pytest.mark.parametrize(
    "recommendations",
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
def test_language_recommendations(recommendations: Recommendations) -> None:
    assert isinstance(recommendations.language_to_citation_candidates, pl.DataFrame)
    assert isinstance(recommendations.citation_to_language, pl.DataFrame)

    assert recommendations.language_to_citation_candidates.shape[1] == 9
    assert recommendations.citation_to_language.shape[1] == 9

    assert all(
        columns
        == [
            "candidate_d3_document_id",
            "cosine_similarity",
            "title",
            "author",
            "publication_date",
            "arxiv_labels",
            "integer_label",
            "semanticscholar_url",
            "arxiv_url",
        ]
        for columns in [
            recommendations.language_to_citation_candidates.columns,
            recommendations.citation_to_language.columns,
        ]
    )

    assert all(
        dtypes
        == [
            pl.Int64,
            pl.Float64,
            pl.Utf8,
            pl.Utf8,
            pl.Utf8,
            pl.List(pl.Utf8),
            pl.Int64,
            pl.Utf8,
            pl.Utf8,
        ]
        for dtypes in [
            recommendations.language_to_citation_candidates.dtypes,
            recommendations.citation_to_language.dtypes,
        ]
    )

    # check that recommendations are sorted in descending order by cosine similarity
    assert recommendations.language_to_citation_candidates["cosine_similarity"].is_sorted(
        descending=True
    )
    assert recommendations.citation_to_language["cosine_similarity"].is_sorted(descending=True)


@pytest.mark.updated
@pytest.mark.parametrize(
    "recommendations",
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
def test_no_missing_values(recommendations: Recommendations) -> None:
    assert recommendations.citation_to_language_candidates.null_count().sum(axis=1).item() == 0
    assert recommendations.citation_to_language.null_count().sum(axis=1).item() == 0
    assert recommendations.language_to_citation_candidates.null_count().sum(axis=1).item() == 0
    assert recommendations.language_to_citation.null_count().sum(axis=1).item() == 0


@pytest.mark.updated
def test_kw_only_initialization_recommendations() -> None:
    with pytest.raises(TypeError):
        Recommendations(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
