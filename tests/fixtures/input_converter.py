import pytest

from readnext.inference import InferenceDataInputConverter
from readnext.utils.aliases import DocumentsFrame


@pytest.fixture
def input_converter(test_documents_frame: DocumentsFrame) -> InferenceDataInputConverter:
    return InferenceDataInputConverter(test_documents_frame)
