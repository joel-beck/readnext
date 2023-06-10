import pytest

from readnext.modeling import (
    DocumentInfo,
    LanguageModelDataConstructor,
    SeenModelDataConstructorPlugin,
    UnseenModelDataConstructorPlugin,
)
from readnext.utils.aliases import (
    DocumentsFrame,
    InfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
    ScoresFrame,
)


@pytest.fixture(scope="session")
def language_model_data_constructor_seen(
    test_documents_frame: DocumentsFrame,
    seen_model_data_constructor_plugin: SeenModelDataConstructorPlugin,
    test_bert_cosine_similarities: ScoresFrame,
) -> LanguageModelDataConstructor:
    query_d3_document_id = 13756489

    return LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=test_documents_frame,
        constructor_plugin=seen_model_data_constructor_plugin,
        cosine_similarity_scores_frame=test_bert_cosine_similarities,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor_unseen(
    test_documents_frame: DocumentsFrame,
    unseen_model_data_constructor_plugin: UnseenModelDataConstructorPlugin,
    test_bert_cosine_similarities: ScoresFrame,
) -> LanguageModelDataConstructor:
    query_d3_document_id = 13756489

    return LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=test_documents_frame,
        constructor_plugin=unseen_model_data_constructor_plugin,
        cosine_similarity_scores_frame=test_bert_cosine_similarities,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor_query_document_seen(
    language_model_data_constructor_seen: LanguageModelDataConstructor,
) -> DocumentInfo:
    return language_model_data_constructor_seen.query_document


@pytest.fixture(scope="session")
def language_model_data_constructor_query_document_unseen(
    language_model_data_constructor_unseen: LanguageModelDataConstructor,
) -> DocumentInfo:
    return language_model_data_constructor_unseen.query_document


@pytest.fixture(scope="session")
def language_model_data_constructor_info_frame_seen(
    language_model_data_constructor_seen: LanguageModelDataConstructor,
) -> InfoFrame:
    return language_model_data_constructor_seen.get_info_frame()


@pytest.fixture(scope="session")
def language_model_data_constructor_info_frame_unseen(
    language_model_data_constructor_unseen: LanguageModelDataConstructor,
) -> InfoFrame:
    return language_model_data_constructor_unseen.get_info_frame()


@pytest.fixture(scope="session")
def language_model_data_constructor_features_frame_seen(
    language_model_data_constructor_seen: LanguageModelDataConstructor,
) -> LanguageFeaturesFrame:
    return language_model_data_constructor_seen.get_features_frame()


@pytest.fixture(scope="session")
def language_model_data_constructor_features_frame_unseen(
    language_model_data_constructor_unseen: LanguageModelDataConstructor,
) -> LanguageFeaturesFrame:
    return language_model_data_constructor_unseen.get_features_frame()


@pytest.fixture(scope="session")
def language_model_data_constructor_integer_labels_frame_seen(
    language_model_data_constructor_seen: LanguageModelDataConstructor,
) -> IntegerLabelsFrame:
    return language_model_data_constructor_seen.get_integer_labels_frame()


@pytest.fixture(scope="session")
def language_model_data_constructor_integer_labels_frame_unseen(
    language_model_data_constructor_unseen: LanguageModelDataConstructor,
) -> IntegerLabelsFrame:
    return language_model_data_constructor_unseen.get_integer_labels_frame()
