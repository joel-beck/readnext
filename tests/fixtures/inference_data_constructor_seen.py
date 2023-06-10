import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import (
    DocumentIdentifier,
    Features,
    InferenceDataConstructor,
    Labels,
    Points,
    Ranks,
    Recommendations,
)
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils.aliases import (
    CitationFeaturesFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    DocumentsFrame,
    InfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
)


@pytest.fixture(scope="session")
def inference_data_constructor_seen_from_semanticscholar_id() -> InferenceDataConstructor:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    return InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
    )


# SECTION: Attribute Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_seen_documents_frame(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> DocumentsFrame:
    return inference_data_constructor_seen_from_semanticscholar_id.documents_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> CitationModelData:
    return inference_data_constructor_seen_from_semanticscholar_id.citation_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_query_document(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> DocumentInfo:
    return inference_data_constructor_seen_citation_model_data.query_document


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_info_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> InfoFrame:
    return inference_data_constructor_seen_citation_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_features_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> CitationFeaturesFrame:
    return inference_data_constructor_seen_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_integer_labels_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> IntegerLabelsFrame:
    return inference_data_constructor_seen_citation_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_ranks_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> CitationRanksFrame:
    return inference_data_constructor_seen_citation_model_data.ranks_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_points_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> CitationPointsFrame:
    return inference_data_constructor_seen_citation_model_data.points_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> LanguageModelData:
    return inference_data_constructor_seen_from_semanticscholar_id.language_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_query_document(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> DocumentInfo:
    return inference_data_constructor_seen_language_model_data.query_document


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_info_frame(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> InfoFrame:
    return inference_data_constructor_seen_language_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_features_frame(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> LanguageFeaturesFrame:
    return inference_data_constructor_seen_language_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_integer_labels(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> IntegerLabelsFrame:
    return inference_data_constructor_seen_language_model_data.integer_labels_frame


# SECTION: Method Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_seen_document_identifier(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> DocumentIdentifier:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_document_identifier()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_document_info(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> DocumentInfo:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_document_info()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_features(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Features:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_features()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_ranks(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Ranks:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_ranks()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_points(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Points:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_points()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_labels(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Labels:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_labels()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_recommendations(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Recommendations:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_recommendations()
