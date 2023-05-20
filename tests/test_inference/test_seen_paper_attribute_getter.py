import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from readnext.inference import DocumentIdentifier, InferenceDataInputConverter
from readnext.inference.attribute_getter import SeenPaperAttributeGetter
from readnext.modeling.language_models.model_choice import LanguageModelChoice

seen_paper_attribute_getters_from_inputs = [
    "seen_paper_attribute_getter_from_semanticscholar_url",
    "seen_paper_attribute_getter_from_semanticscholar_id",
    "seen_paper_attribute_getter_from_arxiv_id",
    "seen_paper_attribute_getter_from_arxiv_url",
]


def test_input_converter_works_for_semanticscholar_id(
    seen_paper_attribute_getter_from_semanticscholar_id: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_from_semanticscholar_id.semanticscholar_id is not None
    assert isinstance(seen_paper_attribute_getter_from_semanticscholar_id.semanticscholar_id, str)
    assert seen_paper_attribute_getter_from_semanticscholar_id.semanticscholar_url is None
    assert seen_paper_attribute_getter_from_semanticscholar_id.arxiv_id is None
    assert seen_paper_attribute_getter_from_semanticscholar_id.arxiv_url is None


def test_input_converter_works_for_semanticscholar_url(
    seen_paper_attribute_getter_from_semanticscholar_url: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_from_semanticscholar_url.semanticscholar_id is None
    assert seen_paper_attribute_getter_from_semanticscholar_url.semanticscholar_url is not None
    assert isinstance(seen_paper_attribute_getter_from_semanticscholar_url.semanticscholar_url, str)
    assert seen_paper_attribute_getter_from_semanticscholar_url.arxiv_id is None
    assert seen_paper_attribute_getter_from_semanticscholar_url.arxiv_url is None


def test_input_converter_works_for_arxiv_id(
    seen_paper_attribute_getter_from_arxiv_id: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_from_arxiv_id.semanticscholar_id is None
    assert seen_paper_attribute_getter_from_arxiv_id.semanticscholar_url is None
    assert seen_paper_attribute_getter_from_arxiv_id.arxiv_id is not None
    assert isinstance(seen_paper_attribute_getter_from_arxiv_id.arxiv_id, str)
    assert seen_paper_attribute_getter_from_arxiv_id.arxiv_url is None


def test_input_converter_works_for_arxiv_url(
    seen_paper_attribute_getter_from_arxiv_url: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_from_arxiv_url.semanticscholar_id is None
    assert seen_paper_attribute_getter_from_arxiv_url.semanticscholar_url is None
    assert seen_paper_attribute_getter_from_arxiv_url.arxiv_id is None
    assert seen_paper_attribute_getter_from_arxiv_url.arxiv_url is not None
    assert isinstance(seen_paper_attribute_getter_from_arxiv_url.arxiv_url, str)


@pytest.mark.parametrize(
    "seen_paper_attribute_getter", lazy_fixture(seen_paper_attribute_getters_from_inputs)
)
def input_converter_sets_identifier_correctly(
    seen_paper_attribute_getter: SeenPaperAttributeGetter,
) -> None:
    assert isinstance(seen_paper_attribute_getter.identifier, DocumentIdentifier)

    assert seen_paper_attribute_getter.identifier.d3_document_id is not None
    assert isinstance(seen_paper_attribute_getter.identifier.d3_document_id, int)

    assert seen_paper_attribute_getter.identifier.semanticscholar_id is not None
    assert isinstance(seen_paper_attribute_getter.identifier.semanticscholar_id, str)

    assert seen_paper_attribute_getter.identifier.semanticscholar_url is not None
    assert isinstance(seen_paper_attribute_getter.identifier.semanticscholar_url, str)

    assert seen_paper_attribute_getter.identifier.arxiv_id is not None
    assert isinstance(seen_paper_attribute_getter.identifier.arxiv_id, str)

    assert seen_paper_attribute_getter.identifier.arxiv_url is not None
    assert isinstance(seen_paper_attribute_getter.identifier.arxiv_url, str)


@pytest.mark.parametrize(
    "seen_paper_attribute_getter", lazy_fixture(seen_paper_attribute_getters_from_inputs)
)
def test_attributes_are_created_correctly(
    seen_paper_attribute_getter: SeenPaperAttributeGetter,
) -> None:
    assert isinstance(
        seen_paper_attribute_getter.language_model_choice,
        LanguageModelChoice,
    )
    assert isinstance(
        seen_paper_attribute_getter.input_converter,
        InferenceDataInputConverter,
    )
    assert isinstance(seen_paper_attribute_getter.documents_data, pd.DataFrame)
    assert isinstance(
        seen_paper_attribute_getter.input_converter.documents_data,
        pd.DataFrame,
    )
    # same data attribute is passed to the input converter
    assert_frame_equal(
        seen_paper_attribute_getter.documents_data,
        seen_paper_attribute_getter.input_converter.documents_data,
    )
