import pandas as pd
import pytest

from readnext.evaluation.scoring import FeatureWeights

# These imports must not come from `readnext.inference`, otherwise they are really
# imported twice with different session scopes and `isinstance()` checks fail.
from readnext.inference import (
    DocumentIdentifier,
    Features,
    InferenceDataConstructor,
    Labels,
    Ranks,
    Recommendations,
)
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils import ScoresFrame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_from_semanticscholar_id() -> InferenceDataConstructor:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    return InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_from_arxiv_url() -> InferenceDataConstructor:
    arxiv_url = "https://arxiv.org/abs/2303.08774"
    return InferenceDataConstructor(
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.scibert,
        feature_weights=FeatureWeights(),
    )


@pytest.fixture(scope="session")
def inference_data_seen_constructor_documents_data(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> pd.DataFrame:
    return inference_data_constructor_seen_from_semanticscholar_id._documents_data


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_documents_data(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> pd.DataFrame:
    return inference_data_unseen_constructor_from_arxiv_url._documents_data


@pytest.fixture(scope="session")
def inference_data_seen_constructor_co_citation_analysis_scores(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_seen_from_semanticscholar_id._co_citation_analysis_scores


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_co_citation_analysis_scores(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_unseen_constructor_from_arxiv_url._co_citation_analysis_scores


@pytest.fixture(scope="session")
def inference_data_seen_constructor_bibliographic_coupling_scores(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_seen_from_semanticscholar_id._bibliographic_coupling_scores


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_bibliographic_coupling_scores(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_unseen_constructor_from_arxiv_url._bibliographic_coupling_scores


@pytest.fixture(scope="session")
def inference_data_seen_constructor_cosine_similarities(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_seen_from_semanticscholar_id._cosine_similarities


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_cosine_similarities(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_unseen_constructor_from_arxiv_url._cosine_similarities


@pytest.fixture(scope="session")
def inference_data_seen_constructor_citation_model_data(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> CitationModelData:
    return inference_data_constructor_seen_from_semanticscholar_id._citation_model_data


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_citation_model_data(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> CitationModelData:
    return inference_data_unseen_constructor_from_arxiv_url._citation_model_data


@pytest.fixture(scope="session")
def inference_data_seen_constructor_language_model_data(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> LanguageModelData:
    return inference_data_constructor_seen_from_semanticscholar_id._language_model_data


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_language_model_data(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> LanguageModelData:
    return inference_data_unseen_constructor_from_arxiv_url._language_model_data


@pytest.fixture(scope="session")
def inference_data_seen_constructor_document_identifier(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> DocumentIdentifier:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_document_identifier()


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_document_identifier(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> DocumentIdentifier:
    return inference_data_unseen_constructor_from_arxiv_url.collect_document_identifier()


@pytest.fixture(scope="session")
def inference_data_seen_constructor_document_info(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> DocumentInfo:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_document_info()


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_document_info(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> DocumentInfo:
    return inference_data_unseen_constructor_from_arxiv_url.collect_document_info()


@pytest.fixture(scope="session")
def inference_data_seen_constructor_features(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Features:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_features()


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_features(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> Features:
    return inference_data_unseen_constructor_from_arxiv_url.collect_features()


@pytest.fixture(scope="session")
def inference_data_seen_constructor_ranks(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Ranks:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_ranks()


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_ranks(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> Ranks:
    return inference_data_unseen_constructor_from_arxiv_url.collect_ranks()


@pytest.fixture(scope="session")
def inference_data_seen_constructor_labels(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Labels:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_labels()


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_labels(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> Labels:
    return inference_data_unseen_constructor_from_arxiv_url.collect_labels()


@pytest.fixture(scope="session")
def inference_data_seen_constructor_recommendations(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Recommendations:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_recommendations()


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_recommendations(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> Recommendations:
    return inference_data_unseen_constructor_from_arxiv_url.collect_recommendations()
