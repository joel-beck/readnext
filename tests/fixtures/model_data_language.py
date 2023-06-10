import pytest

from readnext.modeling import (
    DocumentInfo,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.utils.aliases import InfoFrame, IntegerLabelsFrame, LanguageFeaturesFrame


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


@pytest.fixture(scope="session")
def language_model_data_query_document_seen(
    language_model_data_seen: LanguageModelData,
) -> DocumentInfo:
    return language_model_data_seen.query_document


@pytest.fixture(scope="session")
def language_model_data_query_document_unseen(
    language_model_data_unseen: LanguageModelData,
) -> DocumentInfo:
    return language_model_data_unseen.query_document


@pytest.fixture(scope="session")
def language_model_data_info_frame_seen(
    language_model_data_seen: LanguageModelData,
) -> InfoFrame:
    return language_model_data_seen.info_frame


@pytest.fixture(scope="session")
def language_model_data_info_frame_unseen(
    language_model_data_unseen: LanguageModelData,
) -> InfoFrame:
    return language_model_data_unseen.info_frame


@pytest.fixture(scope="session")
def language_model_data_features_frame_seen(
    language_model_data_seen: LanguageModelData,
) -> LanguageFeaturesFrame:
    return language_model_data_seen.features_frame


@pytest.fixture(scope="session")
def language_model_data_features_frame_unseen(
    language_model_data_unseen: LanguageModelData,
) -> LanguageFeaturesFrame:
    return language_model_data_unseen.features_frame


@pytest.fixture(scope="session")
def language_model_data_integer_labels_seen(
    language_model_data_seen: LanguageModelData,
) -> IntegerLabelsFrame:
    return language_model_data_seen.integer_labels_frame


@pytest.fixture(scope="session")
def language_model_data_integer_labels_unseen(
    language_model_data_unseen: LanguageModelData,
) -> IntegerLabelsFrame:
    return language_model_data_unseen.integer_labels_frame
