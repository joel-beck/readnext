import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

# These imports must not come from `readnext.inference`, otherwise they are really
# imported twice with different session scopes and `isinstance()` checks fail.
from readnext.inference import (
    DocumentIdentifier,
    Features,
    InferenceData,
    InferenceDataConstructor,
    Labels,
    Points,
    Ranks,
    Recommendations,
)
from readnext.modeling import DocumentInfo


@pytest.fixture(scope="session")
def inference_data_seen(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> InferenceData:
    return InferenceData.from_constructor(inference_data_constructor_seen)


@pytest.fixture(scope="session")
def inference_data_unseen(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> InferenceData:
    return InferenceData.from_constructor(inference_data_constructor_unseen)


inference_data_seen_unseen_pair = [
    lazy_fixture("inference_data_seen"),
    lazy_fixture("inference_data_unseen"),
]


@pytest.fixture(scope="session", params=inference_data_seen_unseen_pair)
def inference_data(request: pytest.FixtureRequest) -> InferenceData:
    return request.param


@pytest.fixture(scope="session")
def inference_data_document_identifier_seen(
    inference_data_seen: InferenceData,
) -> DocumentIdentifier:
    return inference_data_seen.document_identifier


@pytest.fixture(scope="session")
def inference_data_document_identifier_unseen(
    inference_data_unseen: InferenceData,
) -> DocumentIdentifier:
    return inference_data_unseen.document_identifier


@pytest.fixture(scope="session")
def inference_data_document_identifier(inference_data: InferenceData) -> DocumentIdentifier:
    return inference_data.document_identifier


@pytest.fixture(scope="session")
def inference_data_document_info_seen(inference_data_seen: InferenceData) -> DocumentInfo:
    return inference_data_seen.document_info


@pytest.fixture(scope="session")
def inference_data_document_info_unseen(inference_data_unseen: InferenceData) -> DocumentInfo:
    return inference_data_unseen.document_info


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
