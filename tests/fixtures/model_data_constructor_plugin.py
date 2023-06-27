import pytest

from readnext.data import SemanticScholarResponse
from readnext.modeling import (
    DocumentInfo,
    SeenModelDataConstructorPlugin,
    UnseenModelDataConstructorPlugin,
)
from readnext.utils.aliases import DocumentsFrame


@pytest.fixture(scope="session")
def model_data_constructor_plugin_seen(
    test_documents_frame: DocumentsFrame,
) -> SeenModelDataConstructorPlugin:
    return SeenModelDataConstructorPlugin(
        d3_document_id=13756489, documents_frame=test_documents_frame
    )


@pytest.fixture(scope="session")
def model_data_constructor_plugin_seen_query_document(
    model_data_constructor_plugin_seen: SeenModelDataConstructorPlugin,
) -> DocumentInfo:
    return model_data_constructor_plugin_seen.collect_query_document()


@pytest.fixture(scope="session")
def model_data_constructor_plugin_unseen(
    semanticscholar_response: SemanticScholarResponse,
) -> UnseenModelDataConstructorPlugin:
    return UnseenModelDataConstructorPlugin(response=semanticscholar_response)


@pytest.fixture(scope="session")
def model_data_constructor_plugin_unseen_query_document(
    model_data_constructor_plugin_unseen: UnseenModelDataConstructorPlugin,
) -> DocumentInfo:
    return model_data_constructor_plugin_unseen.collect_query_document()
