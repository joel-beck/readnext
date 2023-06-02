import polars as pl
import pytest

from readnext.inference import InferenceDataInputConverter


@pytest.fixture
def input_converter_toy_data() -> InferenceDataInputConverter:
    toy_data = pl.DataFrame(
        {
            "d3_document_id": [1001, 1002],
            "semanticscholar_url": [
                "https://www.semanticscholar.org/paper/1",
                "https://www.semanticscholar.org/paper/2",
            ],
            "arxiv_id": ["1", "2"],
        },
    )
    return InferenceDataInputConverter(toy_data)


@pytest.fixture
def input_converter(
    test_documents_data: pl.DataFrame,
) -> InferenceDataInputConverter:
    return InferenceDataInputConverter(test_documents_data)
