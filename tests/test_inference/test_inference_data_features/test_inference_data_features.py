import polars as pl
import pytest
import re
from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier
from readnext.inference.features import Features, Labels, Ranks, Recommendations
from pytest_lazyfixture import lazy_fixture

# TODO: Current Error `recursive dependency involving fixture 'inference_data_features'
# detected` probably because the `inference_data_features` fixture is used in the
# `inference_data_constructor_features` fixture
feature_fixtures = [
    lazy_fixture("inference_data_features"),
    lazy_fixture("inference_data_constructor_features"),
]


@pytest.mark.parametrize("inference_data_features", feature_fixtures)
def test_feature_attributes(inference_data_features: Features) -> None:
    assert isinstance(inference_data_features, Features)
    assert inference_data_features.__dict__.keys() == {
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis",
        "bibliographic_coupling",
        "cosine_similarity",
        "feature_weights",
    }


@pytest.mark.parametrize("inference_data_features", feature_fixtures)
def test_publication_date(inference_data_features: Features) -> None:
    assert isinstance(inference_data_features.publication_date, pl.DataFrame)

    assert inference_data_features.publication_date.shape[1] == 2
    assert inference_data_features.publication_date.columns == [
        "candidate_d3_document_id",
        "publication_date",
    ]

    assert inference_data_features.publication_date["candidate_d3_document_id"].dtype == pl.Int64
    assert inference_data_features.publication_date["publication_date"].dtype == pl.Utf8

    # check that all dates have the format YYYY-MM-DD
    assert all(
        re.match(r"^\d{4}-\d{2}-\d{2}$", date)
        for date in inference_data_features.publication_date["publication_date"]
    )


@pytest.mark.parametrize("inference_data_features", feature_fixtures)
def test_citationcount_document(inference_data_features: Features) -> None:
    assert isinstance(inference_data_features.citationcount_document, pl.DataFrame)

    assert inference_data_features.citationcount_document.shape[1] == 2
    assert inference_data_features.citationcount_document.columns == [
        "candidate_d3_document_id",
        "citationcount_document",
    ]

    assert (
        inference_data_features.citationcount_document["candidate_d3_document_id"].dtype == pl.Int64
    )
    assert (
        inference_data_features.citationcount_document["citationcount_document"].dtype == pl.Int64
    )

    # check that all citation counts are non-negative
    assert all(
        citation_count >= 0
        for citation_count in inference_data_features.citationcount_document[
            "citationcount_document"
        ]
    )


@pytest.mark.parametrize("inference_data_features", feature_fixtures)
def test_citationcount_author(inference_data_features: Features) -> None:
    assert isinstance(inference_data_features.citationcount_author, pl.DataFrame)

    assert inference_data_features.citationcount_author.shape[1] == 2
    assert inference_data_features.citationcount_author.columns == [
        "candidate_d3_document_id",
        "citationcount_author",
    ]

    assert (
        inference_data_features.citationcount_author["candidate_d3_document_id"].dtype == pl.Int64
    )
    assert inference_data_features.citationcount_author["citationcount_author"].dtype == pl.Int64

    # check that all citation counts are non-negative
    assert all(
        citation_count >= 0
        for citation_count in inference_data_features.citationcount_author["citationcount_author"]
    )


@pytest.mark.parametrize("inference_data_features", feature_fixtures)
def test_co_citation_analysis(inference_data_features: Features) -> None:
    assert isinstance(inference_data_features.co_citation_analysis, pl.DataFrame)

    assert inference_data_features.co_citation_analysis.shape[1] == 2
    assert inference_data_features.co_citation_analysis.columns == [
        "candidate_d3_document_id",
        "co_citation_analysis_score",
    ]

    assert (
        inference_data_features.co_citation_analysis["candidate_d3_document_id"].dtype == pl.Int64
    )
    assert (
        inference_data_features.co_citation_analysis["co_citation_analysis_score"].dtype == pl.Int64
    )

    # check that all scores are non-negative
    assert all(
        score >= 0
        for score in inference_data_features.co_citation_analysis["co_citation_analysis_score"]
    )


@pytest.mark.parametrize("inference_data_features", feature_fixtures)
def test_bibliographic_coupling(inference_data_features: Features) -> None:
    assert isinstance(inference_data_features.bibliographic_coupling, pl.DataFrame)

    assert inference_data_features.bibliographic_coupling.shape[1] == 2
    assert inference_data_features.bibliographic_coupling.columns == [
        "candidate_d3_document_id",
        "bibliographic_coupling_score",
    ]

    assert (
        inference_data_features.bibliographic_coupling["candidate_d3_document_id"].dtype == pl.Int64
    )
    assert (
        inference_data_features.bibliographic_coupling["bibliographic_coupling_score"].dtype
        == pl.Int64
    )

    # check that all scores are non-negative
    assert all(
        score >= 0
        for score in inference_data_features.bibliographic_coupling["bibliographic_coupling_score"]
    )


@pytest.mark.parametrize("inference_data_features", feature_fixtures)
def test_cosine_similarity(inference_data_features: Features) -> None:
    assert isinstance(inference_data_features.cosine_similarity, pl.DataFrame)

    assert inference_data_features.cosine_similarity.shape[1] == 2
    assert inference_data_features.cosine_similarity.columns == [
        "candidate_d3_document_id",
        "cosine_similarity",
    ]

    assert inference_data_features.cosine_similarity["candidate_d3_document_id"].dtype == pl.Int64
    assert inference_data_features.cosine_similarity["cosine_similarity"].dtype == pl.Float64

    # check that all scores are between -1 and 1
    assert all(
        inference_data_features.cosine_similarity["cosine_similarity"].is_between(
            lower_bound=-1, upper_bound=1
        )
    )


@pytest.mark.parametrize("inference_data_features", feature_fixtures)
def test_no_missing_values(inference_data_features: Features) -> None:
    assert inference_data_features.publication_date.null_count().sum(axis=1).item() == 0
    assert inference_data_features.citationcount_document.null_count().sum(axis=1).item() == 0
    assert inference_data_features.citationcount_author.null_count().sum(axis=1).item() == 0
    assert inference_data_features.co_citation_analysis.null_count().sum(axis=1).item() == 0
    assert inference_data_features.bibliographic_coupling.null_count().sum(axis=1).item() == 0
    assert inference_data_features.cosine_similarity.null_count().sum(axis=1).item() == 0


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


def test_kw_only_initialization_document_identifier() -> None:
    with pytest.raises(TypeError):
        DocumentIdentifier(
            -1,  # type: ignore
            "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "2303.08774",
            "https://arxiv.org/abs/2303.08774",
        )


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


def test_kw_only_initialization_labels() -> None:
    with pytest.raises(TypeError):
        Labels(pl.DataFrame(), pl.DataFrame())  # type: ignore


def test_kw_only_initialization_recommendations() -> None:
    with pytest.raises(TypeError):
        Recommendations(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
