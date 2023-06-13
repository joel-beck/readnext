import polars as pl
import pytest

# These imports must not come from `readnext.inference`, otherwise they are really
# imported twice with different session scopes and `isinstance()` checks fail.
from readnext.inference import (
    DocumentIdentifier,
    Features,
    InferenceData,
    Labels,
    Points,
    Ranks,
    Recommendations,
)
from readnext.modeling import DocumentInfo
from pytest_lazyfixture import lazy_fixture


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("inference_data_constructor_seen_from_semanticscholar_id"),
        lazy_fixture("inference_data_constructor_unseen_from_arxiv_url"),
    ],
)
def inference_data(request: pytest.FixtureRequest) -> InferenceData:
    return InferenceData.from_constructor(request.param)


@pytest.fixture(scope="session")
def inference_data_document_identifier(inference_data: InferenceData) -> DocumentIdentifier:
    return inference_data.document_identifier


@pytest.fixture(scope="session")
def inference_data_document_info(inference_data: InferenceData) -> DocumentInfo:
    return inference_data.document_info


@pytest.fixture(scope="session")
def inference_data_features(inference_data: InferenceData) -> Features:
    return inference_data.features


@pytest.fixture(scope="session")
def inference_data_ranks(inference_data: InferenceData) -> Ranks:
    return inference_data.ranks


@pytest.fixture(scope="session")
def inference_data_points(inference_data: InferenceData) -> Points:
    return inference_data.points


@pytest.fixture(scope="session")
def inference_data_labels(inference_data: InferenceData) -> Labels:
    return inference_data.labels


@pytest.fixture(scope="session")
def inference_data_recommendations(inference_data: InferenceData) -> Recommendations:
    return inference_data.recommendations


@pytest.fixture(scope="session")
def inference_data_recommendations_citation_to_language(
    inference_data_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_recommendations.citation_to_language


@pytest.fixture(scope="session")
def inference_data_recommendations_citation_to_language_candidates(
    inference_data_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_recommendations.citation_to_language_candidates


@pytest.fixture(scope="session")
def inference_data_recommendations_language_to_citation(
    inference_data_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_recommendations.language_to_citation


@pytest.fixture(scope="session")
def inference_data_recommendations_language_to_citation_candidates(
    inference_data_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_recommendations.language_to_citation_candidates
