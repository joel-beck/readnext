import pandas as pd
import pytest

from readnext.inference import InferenceDataInputConverter
from readnext.utils import (
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
)


@pytest.fixture
def toy_data() -> pd.DataFrame:
    index = pd.Index([1001, 1002], name="document_id", dtype=pd.Int64Dtype())
    return pd.DataFrame(
        {
            "semanticscholar_url": [
                "https://www.semanticscholar.org/paper/1",
                "https://www.semanticscholar.org/paper/2",
            ],
            "arxiv_id": ["1", "2"],
        },
        index=index,
    )


@pytest.fixture
def input_converter_toy_data(toy_data: pd.DataFrame) -> InferenceDataInputConverter:
    return InferenceDataInputConverter(toy_data)


# SECTION: Tests for Toy Data
def test_get_d3_document_id_from_semanticscholar_id_toy_data(
    input_converter_toy_data: InferenceDataInputConverter,
) -> None:
    d3_document_id_1 = input_converter_toy_data.get_d3_document_id_from_semanticscholar_id(
        get_semanticscholar_id_from_semanticscholar_url("https://www.semanticscholar.org/paper/1")
    )
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 1001

    d3_document_id_2 = input_converter_toy_data.get_d3_document_id_from_semanticscholar_id(
        get_semanticscholar_id_from_semanticscholar_url("https://www.semanticscholar.org/paper/2")
    )
    assert isinstance(d3_document_id_2, int)
    assert d3_document_id_2 == 1002

    with pytest.raises(ValueError):
        assert input_converter_toy_data.get_d3_document_id_from_semanticscholar_id(
            get_semanticscholar_id_from_semanticscholar_url(
                "https://www.semanticscholar.org/paper/3"
            )
        )


def test_get_d3_document_id_from_semanticscholar_url_toy_data(
    input_converter_toy_data: InferenceDataInputConverter,
) -> None:
    d3_document_id_1 = input_converter_toy_data.get_d3_document_id_from_semanticscholar_url(
        "https://www.semanticscholar.org/paper/1"
    )
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 1001

    d3_document_id_2 = input_converter_toy_data.get_d3_document_id_from_semanticscholar_url(
        "https://www.semanticscholar.org/paper/2"
    )
    assert isinstance(d3_document_id_2, int)
    assert d3_document_id_2 == 1002

    with pytest.raises(ValueError):
        assert input_converter_toy_data.get_d3_document_id_from_semanticscholar_url(
            "https://www.semanticscholar.org/paper/3"
        )


def test_get_d3_document_id_from_arxiv_id_toy_data(
    input_converter_toy_data: InferenceDataInputConverter,
) -> None:
    d3_document_id_1 = input_converter_toy_data.get_d3_document_id_from_arxiv_id("1")
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 1001

    d3_document_id_2 = input_converter_toy_data.get_d3_document_id_from_arxiv_id("2")
    assert isinstance(d3_document_id_2, int)
    assert d3_document_id_2 == 1002

    with pytest.raises(ValueError):
        assert input_converter_toy_data.get_d3_document_id_from_arxiv_id("3")


def test_get_d3_document_id_from_arxiv_url_toy_data(
    input_converter_toy_data: InferenceDataInputConverter,
) -> None:
    d3_document_id_1 = input_converter_toy_data.get_d3_document_id_from_arxiv_url(
        get_arxiv_url_from_arxiv_id("1")
    )
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 1001

    d3_document_id_2 = input_converter_toy_data.get_d3_document_id_from_arxiv_url(
        get_arxiv_url_from_arxiv_id("2")
    )
    assert isinstance(d3_document_id_2, int)
    assert d3_document_id_2 == 1002

    with pytest.raises(ValueError):
        assert input_converter_toy_data.get_d3_document_id_from_arxiv_url(
            get_arxiv_url_from_arxiv_id("3")
        )


def test_get_semanticscholar_id_from_d3_document_id_toy_data(
    input_converter_toy_data: InferenceDataInputConverter,
) -> None:
    semanticscholar_id_1 = input_converter_toy_data.get_semanticscholar_id_from_d3_document_id(1001)
    assert semanticscholar_id_1 == get_semanticscholar_id_from_semanticscholar_url(
        "https://www.semanticscholar.org/paper/1"
    )
    assert isinstance(semanticscholar_id_1, str)

    semanticscholar_id_2 = input_converter_toy_data.get_semanticscholar_id_from_d3_document_id(1002)
    assert semanticscholar_id_2 == get_semanticscholar_id_from_semanticscholar_url(
        "https://www.semanticscholar.org/paper/2"
    )
    assert isinstance(semanticscholar_id_2, str)

    with pytest.raises(KeyError):
        assert input_converter_toy_data.get_semanticscholar_id_from_d3_document_id(1003)


def test_get_semanticscholar_url_from_d3_document_id_toy_data(
    input_converter_toy_data: InferenceDataInputConverter,
) -> None:
    semanticscholar_url_1 = input_converter_toy_data.get_semanticscholar_url_from_d3_document_id(
        1001
    )
    assert semanticscholar_url_1 == "https://www.semanticscholar.org/paper/1"
    assert isinstance(semanticscholar_url_1, str)

    semanticscholar_url_2 = input_converter_toy_data.get_semanticscholar_url_from_d3_document_id(
        1002
    )
    assert semanticscholar_url_2 == "https://www.semanticscholar.org/paper/2"
    assert isinstance(semanticscholar_url_2, str)

    with pytest.raises(KeyError):
        assert input_converter_toy_data.get_semanticscholar_url_from_d3_document_id(1003)


def test_get_arxiv_id_from_d3_document_id_toy_data(
    input_converter_toy_data: InferenceDataInputConverter,
) -> None:
    arxiv_id_1 = input_converter_toy_data.get_arxiv_id_from_d3_document_id(1001)
    assert arxiv_id_1 == "1"
    assert isinstance(arxiv_id_1, str)

    arxiv_id_2 = input_converter_toy_data.get_arxiv_id_from_d3_document_id(1002)
    assert arxiv_id_2 == "2"
    assert isinstance(arxiv_id_2, str)

    with pytest.raises(KeyError):
        assert input_converter_toy_data.get_arxiv_id_from_d3_document_id(1003)


def test_get_arxiv_url_from_d3_document_id_toy_data(
    input_converter_toy_data: InferenceDataInputConverter,
) -> None:
    arxiv_url_1 = input_converter_toy_data.get_arxiv_url_from_d3_document_id(1001)
    assert arxiv_url_1 == "https://arxiv.org/abs/1"
    assert isinstance(arxiv_url_1, str)

    arxiv_url_2 = input_converter_toy_data.get_arxiv_url_from_d3_document_id(1002)
    assert arxiv_url_2 == "https://arxiv.org/abs/2"
    assert isinstance(arxiv_url_2, str)

    with pytest.raises(KeyError):
        assert input_converter_toy_data.get_arxiv_url_from_d3_document_id(1003)


# SECTION: Tests for Test Data
@pytest.fixture
def input_converter(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> InferenceDataInputConverter:
    return InferenceDataInputConverter(test_documents_authors_labels_citations_most_cited)


def test_get_d3_document_id_from_semanticscholar_id(
    input_converter: InferenceDataInputConverter,
) -> None:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    d3_document_id_1 = input_converter.get_d3_document_id_from_semanticscholar_id(
        semanticscholar_id
    )
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 13756489


def test_get_d3_document_id_from_semanticscholar_url(
    input_converter: InferenceDataInputConverter,
) -> None:
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    d3_document_id_1 = input_converter.get_d3_document_id_from_semanticscholar_url(
        semanticscholar_url
    )
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 13756489


def test_get_d3_document_id_from_arxiv_id(
    input_converter: InferenceDataInputConverter,
) -> None:
    arxiv_id = "1706.03762"
    d3_document_id_1 = input_converter.get_d3_document_id_from_arxiv_id(arxiv_id)
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 13756489


def test_get_d3_document_id_from_arxiv_url(
    input_converter: InferenceDataInputConverter,
) -> None:
    arxiv_url = "https://arxiv.org/abs/1706.03762"
    d3_document_id_1 = input_converter.get_d3_document_id_from_arxiv_url(arxiv_url)
    assert isinstance(d3_document_id_1, int)
    assert d3_document_id_1 == 13756489


def test_get_semanticscholar_id_from_d3_document_id(
    input_converter: InferenceDataInputConverter,
) -> None:
    d3_document_id = 13756489
    semanticscholar_id_1 = input_converter.get_semanticscholar_id_from_d3_document_id(
        d3_document_id
    )
    assert isinstance(semanticscholar_id_1, str)
    assert semanticscholar_id_1 == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"


def test_get_semanticscholar_url_from_d3_document_id(
    input_converter: InferenceDataInputConverter,
) -> None:
    d3_document_id = 13756489
    semanticscholar_url_1 = input_converter.get_semanticscholar_url_from_d3_document_id(
        d3_document_id
    )
    assert isinstance(semanticscholar_url_1, str)
    assert (
        semanticscholar_url_1
        == "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )


def test_get_arxiv_id_from_d3_document_id(
    input_converter: InferenceDataInputConverter,
) -> None:
    d3_document_id = 13756489
    arxiv_id_1 = input_converter.get_arxiv_id_from_d3_document_id(d3_document_id)
    assert isinstance(arxiv_id_1, str)
    assert arxiv_id_1 == "1706.03762"


def test_get_arxiv_url_from_d3_document_id(
    input_converter: InferenceDataInputConverter,
) -> None:
    d3_document_id = 13756489
    arxiv_url_1 = input_converter.get_arxiv_url_from_d3_document_id(d3_document_id)
    assert isinstance(arxiv_url_1, str)
    assert arxiv_url_1 == "https://arxiv.org/abs/1706.03762"
