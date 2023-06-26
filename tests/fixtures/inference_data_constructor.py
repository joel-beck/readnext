import pytest
from pytest_lazyfixture import lazy_fixture

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
    CitationInfoFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    DocumentsFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
    LanguageInfoFrame,
)


@pytest.fixture(scope="session")
def inference_data_constructor_seen() -> InferenceDataConstructor:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    return InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
    )


@pytest.fixture(scope="session")
def inference_data_constructor_unseen() -> InferenceDataConstructor:
    arxiv_url = "https://arxiv.org/abs/2303.08774"
    return InferenceDataConstructor(
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.SCIBERT,
        feature_weights=FeatureWeights(),
    )


# SECTION: Attribute Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_seen_documents_frame(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> DocumentsFrame:
    return inference_data_constructor_seen.documents_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_documents_frame(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> DocumentsFrame:
    return inference_data_constructor_unseen.documents_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> CitationModelData:
    return inference_data_constructor_seen.citation_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> CitationModelData:
    return inference_data_constructor_unseen.citation_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> LanguageModelData:
    return inference_data_constructor_seen.language_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_language_model_data(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> LanguageModelData:
    return inference_data_constructor_unseen.language_model_data


model_data_pair_seen = [
    lazy_fixture("inference_data_constructor_seen_citation_model_data"),
    lazy_fixture("inference_data_constructor_seen_language_model_data"),
]

model_data_pair_unseen = [
    lazy_fixture("inference_data_constructor_unseen_citation_model_data"),
    lazy_fixture("inference_data_constructor_unseen_language_model_data"),
]


@pytest.fixture(scope="session", params=model_data_pair_seen)
def inference_data_constructor_seen_model_data_query_document(
    request: pytest.FixtureRequest,
) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=model_data_pair_unseen)
def inference_data_constructor_unseen_model_data_query_document(
    request: pytest.FixtureRequest,
) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=model_data_pair_seen)
def inference_data_constructor_seen_model_data_integer_labels_frame(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session", params=model_data_pair_unseen)
def inference_data_constructor_unseen_model_data_integer_labels_frame(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_info_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> CitationInfoFrame:
    return inference_data_constructor_seen_citation_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data_info_frame(
    inference_data_constructor_unseen_citation_model_data: CitationModelData,
) -> CitationInfoFrame:
    return inference_data_constructor_unseen_citation_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_info_frame(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> LanguageInfoFrame:
    return inference_data_constructor_seen_language_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_language_model_data_info_frame(
    inference_data_constructor_unseen_language_model_data: LanguageModelData,
) -> LanguageInfoFrame:
    return inference_data_constructor_unseen_language_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_features_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> CitationFeaturesFrame:
    return inference_data_constructor_seen_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data_features_frame(
    inference_data_constructor_unseen_citation_model_data: CitationModelData,
) -> CitationFeaturesFrame:
    return inference_data_constructor_unseen_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_features_frame(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> LanguageFeaturesFrame:
    return inference_data_constructor_seen_language_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_language_model_data_features_frame(
    inference_data_constructor_unseen_language_model_data: LanguageModelData,
) -> LanguageFeaturesFrame:
    return inference_data_constructor_unseen_language_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_ranks_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> CitationRanksFrame:
    return inference_data_constructor_seen_citation_model_data.ranks_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data_ranks_frame(
    inference_data_constructor_unseen_citation_model_data: CitationModelData,
) -> CitationRanksFrame:
    return inference_data_constructor_unseen_citation_model_data.ranks_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_points_frame(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> CitationPointsFrame:
    return inference_data_constructor_seen_citation_model_data.points_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data_points_frame(
    inference_data_constructor_unseen_citation_model_data: CitationModelData,
) -> CitationPointsFrame:
    return inference_data_constructor_unseen_citation_model_data.points_frame


# SECTION: Method Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_seen_document_identifier(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> DocumentIdentifier:
    return inference_data_constructor_seen.collect_document_identifier()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_document_identifier(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> DocumentIdentifier:
    return inference_data_constructor_unseen.collect_document_identifier()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_document_info(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> DocumentInfo:
    return inference_data_constructor_seen.collect_document_info()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_document_info(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> DocumentInfo:
    return inference_data_constructor_unseen.collect_document_info()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_labels(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> Labels:
    return inference_data_constructor_seen.collect_labels()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_labels(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> Labels:
    return inference_data_constructor_unseen.collect_labels()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_features(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> Features:
    return inference_data_constructor_seen.collect_features()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_features(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> Features:
    return inference_data_constructor_unseen.collect_features()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_ranks(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> Ranks:
    return inference_data_constructor_seen.collect_ranks()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_ranks(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> Ranks:
    return inference_data_constructor_unseen.collect_ranks()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_points(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> Points:
    return inference_data_constructor_seen.collect_points()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_points(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> Points:
    return inference_data_constructor_unseen.collect_points()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_recommendations(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> Recommendations:
    return inference_data_constructor_seen.collect_recommendations()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_recommendations(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> Recommendations:
    return inference_data_constructor_unseen.collect_recommendations()
