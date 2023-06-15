import polars as pl
import pytest

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


@pytest.fixture(scope="session")
def inference_data_seen_document_identifier(
    inference_data_seen: InferenceData,
) -> DocumentIdentifier:
    return inference_data_seen.document_identifier


@pytest.fixture(scope="session")
def inference_data_unseen_document_identifier(
    inference_data_unseen: InferenceData,
) -> DocumentIdentifier:
    return inference_data_unseen.document_identifier


@pytest.fixture(scope="session")
def inference_data_seen_document_info(inference_data_seen: InferenceData) -> DocumentInfo:
    return inference_data_seen.document_info


@pytest.fixture(scope="session")
def inference_data_unseen_document_info(inference_data_unseen: InferenceData) -> DocumentInfo:
    return inference_data_unseen.document_info


@pytest.fixture(scope="session")
def inference_data_seen_features(inference_data_seen: InferenceData) -> Features:
    return inference_data_seen.features


@pytest.fixture(scope="session")
def inference_data_unseen_features(inference_data_unseen: InferenceData) -> Features:
    return inference_data_unseen.features


@pytest.fixture(scope="session")
def inference_data_seen_ranks(inference_data_seen: InferenceData) -> Ranks:
    return inference_data_seen.ranks


@pytest.fixture(scope="session")
def inference_data_unseen_ranks(inference_data_unseen: InferenceData) -> Ranks:
    return inference_data_unseen.ranks


@pytest.fixture(scope="session")
def inference_data_seen_points(inference_data_seen: InferenceData) -> Points:
    return inference_data_seen.points


@pytest.fixture(scope="session")
def inference_data_unseen_points(inference_data_unseen: InferenceData) -> Points:
    return inference_data_unseen.points


@pytest.fixture(scope="session")
def inference_data_seen_labels(inference_data_seen: InferenceData) -> Labels:
    return inference_data_seen.labels


@pytest.fixture(scope="session")
def inference_data_unseen_labels(inference_data_unseen: InferenceData) -> Labels:
    return inference_data_unseen.labels


@pytest.fixture(scope="session")
def inference_data_seen_recommendations(inference_data_seen: InferenceData) -> Recommendations:
    return inference_data_seen.recommendations


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations(inference_data_unseen: InferenceData) -> Recommendations:
    return inference_data_unseen.recommendations


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_citation_to_language(
    inference_data_seen_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_seen_recommendations.citation_to_language


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_citation_to_language(
    inference_data_unseen_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_unseen_recommendations.citation_to_language


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_citation_to_language_candidates(
    inference_data_seen_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_seen_recommendations.citation_to_language_candidates


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_citation_to_language_candidates(
    inference_data_unseen_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_unseen_recommendations.citation_to_language_candidates


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_language_to_citation(
    inference_data_seen_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_seen_recommendations.language_to_citation


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_language_to_citation(
    inference_data_unseen_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_unseen_recommendations.language_to_citation


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_language_to_citation_candidates(
    inference_data_seen_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_seen_recommendations.language_to_citation_candidates


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_language_to_citation_candidates(
    inference_data_unseen_recommendations: Recommendations,
) -> pl.DataFrame:
    return inference_data_unseen_recommendations.language_to_citation_candidates
