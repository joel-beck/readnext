import pytest

from readnext.data import SemanticScholarResponse
from readnext.modeling import SeenModelDataConstructorPlugin, UnseenModelDataConstructorPlugin
from readnext.utils.aliases import DocumentsFrame


@pytest.fixture(scope="session")
def seen_model_data_constructor_plugin(
    test_documents_frame: DocumentsFrame,
) -> SeenModelDataConstructorPlugin:
    return SeenModelDataConstructorPlugin(d3_document_id=-1, documents_frame=test_documents_frame)


@pytest.fixture(scope="session")
def unseen_model_data_constructor_plugin(
    semanticscholar_response: SemanticScholarResponse,
) -> UnseenModelDataConstructorPlugin:
    return UnseenModelDataConstructorPlugin(response=semanticscholar_response)
