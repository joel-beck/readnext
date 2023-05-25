import pandas as pd
import pytest

# These imports must not come from `readnext.inference`, otherwise they are really
# imported twice with different session scopes and `isinstance()` checks fail.
from readnext.inference import (
    DocumentIdentifier,
    Features,
    InferenceData,
    InferenceDataConstructor,
    Labels,
    Ranks,
    Recommendations,
)
from readnext.modeling import DocumentInfo


@pytest.fixture(scope="session")
def inference_data_seen_from_semanticscholar_id(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> InferenceData:
    return InferenceData.from_constructor(inference_data_constructor_seen_from_semanticscholar_id)


@pytest.fixture(scope="session")
def inference_data_unseen_from_arxiv_url(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> InferenceData:
    return InferenceData.from_constructor(inference_data_unseen_constructor_from_arxiv_url)


@pytest.fixture(scope="session")
def inference_data_seen_document_identifier(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> DocumentIdentifier:
    return inference_data_seen_from_semanticscholar_id.document_identifier


@pytest.fixture(scope="session")
def inference_data_unseen_document_identifier(
    inference_data_unseen_from_arxiv_url: InferenceData,
) -> DocumentIdentifier:
    return inference_data_unseen_from_arxiv_url.document_identifier


@pytest.fixture(scope="session")
def inference_data_seen_document_info(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> DocumentInfo:
    return inference_data_seen_from_semanticscholar_id.document_info


@pytest.fixture(scope="session")
def inference_data_unseen_document_info(
    inference_data_unseen_from_arxiv_url: InferenceData,
) -> DocumentInfo:
    return inference_data_unseen_from_arxiv_url.document_info


@pytest.fixture(scope="session")
def inference_data_seen_features(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> Features:
    return inference_data_seen_from_semanticscholar_id.features


@pytest.fixture(scope="session")
def inference_data_unseen_features(inference_data_unseen_from_arxiv_url: InferenceData) -> Features:
    return inference_data_unseen_from_arxiv_url.features


@pytest.fixture(scope="session")
def inference_data_seen_ranks(inference_data_seen_from_semanticscholar_id: InferenceData) -> Ranks:
    return inference_data_seen_from_semanticscholar_id.ranks


@pytest.fixture(scope="session")
def inference_data_unseen_ranks(inference_data_unseen_from_arxiv_url: InferenceData) -> Ranks:
    return inference_data_unseen_from_arxiv_url.ranks


@pytest.fixture(scope="session")
def inference_data_seen_labels(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> Labels:
    return inference_data_seen_from_semanticscholar_id.labels


@pytest.fixture(scope="session")
def inference_data_unseen_labels(inference_data_unseen_from_arxiv_url: InferenceData) -> Labels:
    return inference_data_unseen_from_arxiv_url.labels


@pytest.fixture(scope="session")
def inference_data_seen_recommendations(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> Recommendations:
    return inference_data_seen_from_semanticscholar_id.recommendations


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_citation_to_language(
    inference_data_seen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_seen_recommendations.citation_to_language


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_citation_to_language_candidates(
    inference_data_seen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_seen_recommendations.citation_to_language_candidates


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_language_to_citation(
    inference_data_seen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_seen_recommendations.language_to_citation


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_language_to_citation_candidates(
    inference_data_seen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_seen_recommendations.language_to_citation_candidates


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations(
    inference_data_unseen_from_arxiv_url: InferenceData,
) -> Recommendations:
    return inference_data_unseen_from_arxiv_url.recommendations


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_citation_to_language(
    inference_data_unseen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_unseen_recommendations.citation_to_language


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_citation_to_language_candidates(
    inference_data_unseen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_unseen_recommendations.citation_to_language_candidates


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_language_to_citation(
    inference_data_unseen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_unseen_recommendations.language_to_citation


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_language_to_citation_candidates(
    inference_data_unseen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_unseen_recommendations.language_to_citation_candidates
