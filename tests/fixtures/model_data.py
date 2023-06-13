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
    InfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
)


@pytest.fixture(scope="session")
def citation_model_data(
    request: pytest.FixtureRequest,
    citation_model_data_constructor_seen: CitationModelDataConstructor,
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
) -> CitationModelData:
    if request.param == "seen":
        return CitationModelData.from_constructor(citation_model_data_constructor_seen)
    if request.param == "unseen":
        return CitationModelData.from_constructor(citation_model_data_constructor_unseen)

    raise ValueError(f"Invalid parameter value: {request.param}")


@pytest.fixture(scope="session", params=["seen", "unseen"])
def language_model_data(
    request: pytest.FixtureRequest,
    language_model_data_constructor_seen: LanguageModelDataConstructor,
    language_model_data_constructor_unseen: LanguageModelDataConstructor,
) -> LanguageModelData:
    if request.param == "seen":
        return LanguageModelData.from_constructor(language_model_data_constructor_seen)
    if request.param == "unseen":
        return LanguageModelData.from_constructor(language_model_data_constructor_unseen)

    raise ValueError(f"Invalid parameter value: {request.param}")


@pytest.fixture(
    scope="session",
    params=[lazy_fixture("citation_model_data"), lazy_fixture("language_model_data")],
)
def model_data_query_document(request: pytest.FixtureRequest) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(
    scope="session",
    params=[lazy_fixture("citation_model_data"), lazy_fixture("language_model_data")],
)
def model_data_info_frame(request: pytest.FixtureRequest) -> InfoFrame:
    return request.param.info_frame


@pytest.fixture(
    scope="session",
    params=[lazy_fixture("citation_model_data"), lazy_fixture("language_model_data")],
)
def model_data_integer_labels(request: pytest.FixtureRequest) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session")
def citation_model_data_features_frame(
    citation_model_data: CitationModelData,
) -> CitationFeaturesFrame:
    return citation_model_data.features_frame


@pytest.fixture(scope="session")
def language_model_data_features_frame(
    language_model_data: LanguageModelData,
) -> LanguageFeaturesFrame:
    return language_model_data.features_frame


@pytest.fixture(scope="session")
def citation_model_data_ranks_frame(citation_model_data: CitationModelData) -> CitationRanksFrame:
    return citation_model_data.ranks_frame


@pytest.fixture(scope="session")
def citation_model_data_points_frame(citation_model_data: CitationModelData) -> CitationPointsFrame:
    return citation_model_data.points_frame
