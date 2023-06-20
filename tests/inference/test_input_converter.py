import pytest
from polars.exceptions import ComputeError

from readnext.inference import InferenceDataInputConverter


@pytest.mark.updated
def test_get_d3_document_id_from_semanticscholar_id(
    inference_data_input_converter: InferenceDataInputConverter,
) -> None:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    d3_document_id_1 = inference_data_input_converter.get_d3_document_id_from_semanticscholar_id(
        semanticscholar_id
    )
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 13756489


@pytest.mark.updated
def test_get_d3_document_id_from_semanticscholar_url(
    inference_data_input_converter: InferenceDataInputConverter,
) -> None:
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    d3_document_id_1 = inference_data_input_converter.get_d3_document_id_from_semanticscholar_url(
        semanticscholar_url
    )
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 13756489


@pytest.mark.updated
def test_get_d3_document_id_from_arxiv_id(
    inference_data_input_converter: InferenceDataInputConverter,
) -> None:
    arxiv_id = "1706.03762"
    d3_document_id_1 = inference_data_input_converter.get_d3_document_id_from_arxiv_id(arxiv_id)
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 13756489


@pytest.mark.updated
def test_get_d3_document_id_from_arxiv_url(
    inference_data_input_converter: InferenceDataInputConverter,
) -> None:
    arxiv_url = "https://arxiv.org/abs/1706.03762"
    d3_document_id_1 = inference_data_input_converter.get_d3_document_id_from_arxiv_url(arxiv_url)
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 13756489


@pytest.mark.updated
def test_get_semanticscholar_id_from_d3_document_id(
    inference_data_input_converter: InferenceDataInputConverter,
) -> None:
    # Test valid input
    d3_document_id = 13756489
    semanticscholar_id_1 = (
        inference_data_input_converter.get_semanticscholar_id_from_d3_document_id(d3_document_id)
    )
    assert isinstance(semanticscholar_id_1, str)
    assert semanticscholar_id_1 == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    # Test invalid input
    d3_document_id = "invalid_id"
    with pytest.raises(ComputeError):
        inference_data_input_converter.get_semanticscholar_id_from_d3_document_id(
            d3_document_id  # type: ignore
        )


@pytest.mark.updated
def test_get_semanticscholar_url_from_d3_document_id(
    inference_data_input_converter: InferenceDataInputConverter,
) -> None:
    # Test valid input
    d3_document_id = 13756489
    semanticscholar_url_1 = (
        inference_data_input_converter.get_semanticscholar_url_from_d3_document_id(d3_document_id)
    )
    assert isinstance(semanticscholar_url_1, str)
    assert (
        semanticscholar_url_1
        == "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )

    # Test invalid input
    d3_document_id = "invalid_id"
    with pytest.raises(ComputeError):
        inference_data_input_converter.get_semanticscholar_url_from_d3_document_id(
            d3_document_id  # type: ignore
        )


@pytest.mark.updated
def test_get_arxiv_id_from_d3_document_id(
    inference_data_input_converter: InferenceDataInputConverter,
) -> None:
    # Test valid input
    d3_document_id = 13756489
    arxiv_id_1 = inference_data_input_converter.get_arxiv_id_from_d3_document_id(d3_document_id)
    assert isinstance(arxiv_id_1, str)
    assert arxiv_id_1 == "1706.03762"

    # Test invalid input
    d3_document_id = "invalid_id"
    with pytest.raises(ComputeError):
        inference_data_input_converter.get_arxiv_id_from_d3_document_id(
            d3_document_id  # type: ignore
        )


@pytest.mark.updated
def test_get_arxiv_url_from_d3_document_id(
    inference_data_input_converter: InferenceDataInputConverter,
) -> None:
    # Test valid input
    d3_document_id = 13756489
    arxiv_url_1 = inference_data_input_converter.get_arxiv_url_from_d3_document_id(d3_document_id)
    assert isinstance(arxiv_url_1, str)
    assert arxiv_url_1 == "https://arxiv.org/abs/1706.03762"

    # Test invalid input
    d3_document_id = "invalid_id"
    with pytest.raises(ComputeError):
        inference_data_input_converter.get_arxiv_url_from_d3_document_id(
            d3_document_id  # type: ignore
        )
