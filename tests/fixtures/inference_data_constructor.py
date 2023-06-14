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
    CitationPointsFrame,
    CitationRanksFrame,
    DocumentsFrame,
    InfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
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
constructor_seen_unseen_pair = [
    lazy_fixture("inference_data_constructor_seen"),
    lazy_fixture("inference_data_constructor_unseen"),
]


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_documents_frame(
    request: pytest.FixtureRequest,
) -> DocumentsFrame:
    return request.param.documents_frame


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_citation_model_data(
    request: pytest.FixtureRequest,
) -> CitationModelData:
    return request.param.citation_model_data


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_language_model_data(
    request: pytest.FixtureRequest,
) -> LanguageModelData:
    return request.param.language_model_data


model_data_pair = [
    lazy_fixture("inference_data_constructor_citation_model_data"),
    lazy_fixture("inference_data_constructor_language_model_data"),
]


@pytest.fixture(scope="session", params=model_data_pair)
def inference_data_constructor_model_data_query_document(
    request: pytest.FixtureRequest,
) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=model_data_pair)
def inference_data_constructor_model_data_info_frame(
    request: pytest.FixtureRequest,
) -> InfoFrame:
    return request.param.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_model_data_integer_labels_frame(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_citation_model_data_features_frame(
    inference_data_constructor_citation_model_data: CitationModelData,
) -> CitationFeaturesFrame:
    return inference_data_constructor_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_features_frame(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> LanguageFeaturesFrame:
    return inference_data_constructor_seen_language_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_citation_model_data_ranks_frame(
    inference_data_constructor_citation_model_data: CitationModelData,
) -> CitationRanksFrame:
    return inference_data_constructor_citation_model_data.ranks_frame


@pytest.fixture(scope="session")
def inference_data_constructor_citation_model_data_points_frame(
    inference_data_constructor_citation_model_data: CitationModelData,
) -> CitationPointsFrame:
    return inference_data_constructor_citation_model_data.points_frame


# SECTION: Method Fixtures
@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_document_identifier(
    request: pytest.FixtureRequest,
) -> DocumentIdentifier:
    return request.param.collect_document_identifier()


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_document_info(request: pytest.FixtureRequest) -> DocumentInfo:
    return request.param.collect_document_info()


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_labels(request: pytest.FixtureRequest) -> Labels:
    return request.param.collect_labels()


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_features(request: pytest.FixtureRequest) -> Features:
    return request.param.collect_features()


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_ranks(request: pytest.FixtureRequest) -> Ranks:
    return request.param.collect_ranks()


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_points(request: pytest.FixtureRequest) -> Points:
    return request.param.collect_points()


@pytest.fixture(scope="session", params=constructor_seen_unseen_pair)
def inference_data_constructor_recommendations(request: pytest.FixtureRequest) -> Recommendations:
    return request.param.collect_recommendations()
