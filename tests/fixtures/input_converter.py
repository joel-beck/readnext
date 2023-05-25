import pandas as pd
import pytest

from readnext.inference import InferenceDataInputConverter


@pytest.fixture
def input_converter_toy_data() -> InferenceDataInputConverter:
    index = pd.Index([1001, 1002], name="document_id", dtype=pd.Int64Dtype())
    toy_data = pd.DataFrame(
        {
            "semanticscholar_url": [
                "https://www.semanticscholar.org/paper/1",
                "https://www.semanticscholar.org/paper/2",
            ],
            "arxiv_id": ["1", "2"],
        },
        index=index,
    )
    return InferenceDataInputConverter(toy_data)


@pytest.fixture
def input_converter(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> InferenceDataInputConverter:
    return InferenceDataInputConverter(test_documents_authors_labels_citations_most_cited)
