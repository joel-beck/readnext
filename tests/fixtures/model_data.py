import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.utils.aliases import (
    CitationFeaturesFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    CitationInfoFrame,
    LanguageInfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
)


@pytest.fixture(scope="session")
def citation_model_data_seen(
    citation_model_data_constructor_seen: CitationModelDataConstructor,
) -> CitationModelData:
    return CitationModelData.from_constructor(citation_model_data_constructor_seen)


@pytest.fixture(scope="session")
def citation_model_data_unseen(
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
) -> CitationModelData:
    return CitationModelData.from_constructor(citation_model_data_constructor_unseen)


@pytest.fixture(scope="session")
def language_model_data_seen(
    language_model_data_constructor_seen: LanguageModelDataConstructor,
) -> LanguageModelData:
    return LanguageModelData.from_constructor(language_model_data_constructor_seen)


@pytest.fixture(scope="session")
def language_model_data_unseen(
    language_model_data_constructor_unseen: LanguageModelDataConstructor,
) -> LanguageModelData:
    return LanguageModelData.from_constructor(language_model_data_constructor_unseen)


citation_model_data = [
    lazy_fixture("citation_model_data_seen"),
    lazy_fixture("citation_model_data_unseen"),
]

language_model_data = [
    lazy_fixture("language_model_data_seen"),
    lazy_fixture("language_model_data_unseen"),
]

seen_model_data = [
    lazy_fixture("citation_model_data_seen"),
    lazy_fixture("language_model_data_seen"),
]

unseen_model_data = [
    lazy_fixture("citation_model_data_unseen"),
    lazy_fixture("language_model_data_unseen"),
]

model_data = citation_model_data + language_model_data


@pytest.fixture(scope="session", params=seen_model_data)
def model_data_seen_query_document(request: pytest.FixtureRequest) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=unseen_model_data)
def model_data_unseen_query_document(request: pytest.FixtureRequest) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=seen_model_data)
def model_data_seen_integer_labels_frame(request: pytest.FixtureRequest) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session", params=unseen_model_data)
def model_data_unseen_integer_labels_frame(request: pytest.FixtureRequest) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session", params=citation_model_data)
def citation_model_data_info_frame(
    request: pytest.FixtureRequest,
) -> CitationInfoFrame:
    return request.param.info_frame


@pytest.fixture(scope="session", params=language_model_data)
def language_model_data_info_frame(
    request: pytest.FixtureRequest,
) -> LanguageInfoFrame:
    return request.param.info_frame


@pytest.fixture(scope="session", params=citation_model_data)
def citation_model_data_features_frame(
    request: pytest.FixtureRequest,
) -> CitationFeaturesFrame:
    return request.param.features_frame


@pytest.fixture(scope="session", params=language_model_data)
def language_model_data_features_frame(
    request: pytest.FixtureRequest,
) -> LanguageFeaturesFrame:
    return request.param.features_frame


@pytest.fixture(scope="session", params=citation_model_data)
def citation_model_data_ranks_frame(
    request: pytest.FixtureRequest,
) -> CitationRanksFrame:
    return request.param.ranks_frame


@pytest.fixture(scope="session", params=citation_model_data)
def citation_model_data_points_frame(
    request: pytest.FixtureRequest,
) -> CitationPointsFrame:
    return request.param.points_frame
