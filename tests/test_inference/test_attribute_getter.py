import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from readnext.data.semanticscholar import (
    SemanticScholarCitation,
    SemanticScholarReference,
    SemanticscholarRequest,
    SemanticScholarResponse,
)
from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, InferenceDataInputConverter
from readnext.inference.attribute_getter import (
    AttributeGetter,
    SeenPaperAttributeGetter,
    UnseenPaperAttributeGetter,
)
from readnext.modeling.document_info import DocumentInfo
from readnext.modeling.language_models import LanguageModelChoice

seen_paper_attribute_getters = [
    "seen_paper_attribute_getter_from_semanticscholar_url",
    "seen_paper_attribute_getter_from_semanticscholar_id",
    "seen_paper_attribute_getter_from_arxiv_id",
    "seen_paper_attribute_getter_from_arxiv_url",
]

unseen_paper_attribute_getters = [
    "unseen_paper_attribute_getter_from_semanticscholar_url",
    "unseen_paper_attribute_getter_from_semanticscholar_id",
    "unseen_paper_attribute_getter_from_arxiv_id",
    "unseen_paper_attribute_getter_from_arxiv_url",
]

attribute_getters = seen_paper_attribute_getters + unseen_paper_attribute_getters


# SECTION: AttributeGetter
@pytest.mark.parametrize("attribute_getter", lazy_fixture(attribute_getters))
def test_identifier_is_set_correctly(
    attribute_getter: AttributeGetter,
) -> None:
    assert attribute_getter.identifier.d3_document_id is not None
    assert isinstance(attribute_getter.identifier.d3_document_id, int)

    assert attribute_getter.identifier.semanticscholar_id is not None
    assert isinstance(attribute_getter.identifier.semanticscholar_id, str)

    assert attribute_getter.identifier.semanticscholar_url is not None
    assert isinstance(attribute_getter.identifier.semanticscholar_url, str)

    assert attribute_getter.identifier.arxiv_id is not None
    assert isinstance(attribute_getter.identifier.arxiv_id, str)

    assert attribute_getter.identifier.arxiv_url is not None
    assert isinstance(attribute_getter.identifier.arxiv_url, str)


@pytest.mark.parametrize("attribute_getter", lazy_fixture(attribute_getters))
def test_attribute_getter_attributes_are_created_correctly(
    attribute_getter: AttributeGetter,
) -> None:
    assert isinstance(attribute_getter.identifier, DocumentIdentifier)

    assert isinstance(attribute_getter.language_model_choice, LanguageModelChoice)

    assert isinstance(attribute_getter.feature_weights, FeatureWeights)

    assert isinstance(attribute_getter.documents_data, pd.DataFrame)


# SECTION: SeenPaperAttributeGetter
def test_passing_inputs_to_input_converter_works_for_semanticscholar_id(
    seen_paper_attribute_getter_from_semanticscholar_id: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_from_semanticscholar_id.semanticscholar_id is not None
    assert isinstance(seen_paper_attribute_getter_from_semanticscholar_id.semanticscholar_id, str)
    assert seen_paper_attribute_getter_from_semanticscholar_id.semanticscholar_url is None
    assert seen_paper_attribute_getter_from_semanticscholar_id.arxiv_id is None
    assert seen_paper_attribute_getter_from_semanticscholar_id.arxiv_url is None


def test_passing_inputs_to_input_converter_works_for_semanticscholar_url(
    seen_paper_attribute_getter_from_semanticscholar_url: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_from_semanticscholar_url.semanticscholar_id is None
    assert seen_paper_attribute_getter_from_semanticscholar_url.semanticscholar_url is not None
    assert isinstance(seen_paper_attribute_getter_from_semanticscholar_url.semanticscholar_url, str)
    assert seen_paper_attribute_getter_from_semanticscholar_url.arxiv_id is None
    assert seen_paper_attribute_getter_from_semanticscholar_url.arxiv_url is None


def test_passing_inputs_to_input_converter_works_for_arxiv_id(
    seen_paper_attribute_getter_from_arxiv_id: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_from_arxiv_id.semanticscholar_id is None
    assert seen_paper_attribute_getter_from_arxiv_id.semanticscholar_url is None
    assert seen_paper_attribute_getter_from_arxiv_id.arxiv_id is not None
    assert isinstance(seen_paper_attribute_getter_from_arxiv_id.arxiv_id, str)
    assert seen_paper_attribute_getter_from_arxiv_id.arxiv_url is None


def test_passing_inputs_to_input_converter_works_for_arxiv_url(
    seen_paper_attribute_getter_from_arxiv_url: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_from_arxiv_url.semanticscholar_id is None
    assert seen_paper_attribute_getter_from_arxiv_url.semanticscholar_url is None
    assert seen_paper_attribute_getter_from_arxiv_url.arxiv_id is None
    assert seen_paper_attribute_getter_from_arxiv_url.arxiv_url is not None
    assert isinstance(seen_paper_attribute_getter_from_arxiv_url.arxiv_url, str)


@pytest.mark.parametrize("seen_paper_attribute_getter", lazy_fixture(seen_paper_attribute_getters))
def test_seen_paper_attribute_getter_attributes_are_created_correctly(
    seen_paper_attribute_getter: SeenPaperAttributeGetter,
) -> None:
    assert isinstance(seen_paper_attribute_getter.input_converter, InferenceDataInputConverter)

    assert isinstance(seen_paper_attribute_getter.input_converter, InferenceDataInputConverter)

    assert isinstance(seen_paper_attribute_getter.input_converter.documents_data, pd.DataFrame)

    # same data attribute is passed to the input converter
    assert_frame_equal(
        seen_paper_attribute_getter.documents_data,
        seen_paper_attribute_getter.input_converter.documents_data,
    )


# SECTION: UnseenPaperAttributeGetter
@pytest.mark.parametrize(
    "unseen_paper_attribute_getter", lazy_fixture(unseen_paper_attribute_getters)
)
def test_unseen_paper_attribute_getter_attributes_are_created_correctly(
    unseen_paper_attribute_getter: UnseenPaperAttributeGetter,
) -> None:
    assert isinstance(unseen_paper_attribute_getter.semanticscholar_request, SemanticscholarRequest)

    assert isinstance(unseen_paper_attribute_getter.response, SemanticScholarResponse)


@pytest.mark.parametrize(
    "unseen_paper_attribute_getter", lazy_fixture(unseen_paper_attribute_getters)
)
def test_send_semanticscholar_request(
    unseen_paper_attribute_getter: UnseenPaperAttributeGetter,
) -> None:
    unseen_paper_attribute_getter.send_semanticscholar_request()

    assert isinstance(unseen_paper_attribute_getter.response, SemanticScholarResponse)

    # defaults are empty strings instead of None => check for length > 0 for set
    # attributes
    assert isinstance(unseen_paper_attribute_getter.response.semanticscholar_id, str)
    assert len(unseen_paper_attribute_getter.response.semanticscholar_id) > 0

    assert isinstance(unseen_paper_attribute_getter.response.arxiv_id, str)

    assert isinstance(unseen_paper_attribute_getter.response.title, str)
    assert len(unseen_paper_attribute_getter.response.title) > 0

    assert isinstance(unseen_paper_attribute_getter.response.abstract, str)
    assert len(unseen_paper_attribute_getter.response.abstract) > 0

    assert isinstance(unseen_paper_attribute_getter.response.citations, list)
    assert all(
        isinstance(citation, dict) for citation in unseen_paper_attribute_getter.response.citations
    )
    # if paperId or title is set, it is a string with length > 0
    assert all(
        (isinstance(citation["paperId"], str) and len(citation["paperId"]) > 0)
        or citation["paperId"] is None
        for citation in unseen_paper_attribute_getter.response.citations
    )
    assert all(
        (isinstance(citation["title"], str) and len(citation["title"]) > 0)
        or citation["title"] is None
        for citation in unseen_paper_attribute_getter.response.citations
    )

    assert unseen_paper_attribute_getter.response.references is not None
    assert isinstance(unseen_paper_attribute_getter.response.references, list)
    assert all(
        isinstance(reference, dict)
        for reference in unseen_paper_attribute_getter.response.references
    )
    assert all(
        (isinstance(citation["paperId"], str) and len(citation["paperId"]) > 0)
        or citation["paperId"] is None
        for citation in unseen_paper_attribute_getter.response.references
    )
    assert all(
        (isinstance(citation["title"], str) and len(citation["title"]) > 0)
        or citation["title"] is None
        for citation in unseen_paper_attribute_getter.response.references
    )


@pytest.mark.parametrize(
    "unseen_paper_attribute_getter", lazy_fixture(unseen_paper_attribute_getters)
)
def test_query_citation_urls(
    unseen_paper_attribute_getter: UnseenPaperAttributeGetter,
) -> None:
    citation_urls = unseen_paper_attribute_getter.get_query_citation_urls()

    assert isinstance(citation_urls, list)
    assert all(isinstance(citation_url, str) for citation_url in citation_urls)

    # check that all set citation urls are semanticscholar urls
    assert all(
        citation_url.startswith("https://www.semanticscholar.org/paper/")
        for citation_url in citation_urls
        if len(citation_url) > 0
    )


@pytest.mark.parametrize(
    "unseen_paper_attribute_getter", lazy_fixture(unseen_paper_attribute_getters)
)
def test_query_reference_urls(
    unseen_paper_attribute_getter: UnseenPaperAttributeGetter,
) -> None:
    reference_urls = unseen_paper_attribute_getter.get_query_reference_urls()

    assert isinstance(reference_urls, list)
    assert all(isinstance(reference_url, str) for reference_url in reference_urls)

    # check that all set reference urls are semanticscholar urls
    assert all(
        reference_url.startswith("https://www.semanticscholar.org/paper/")
        for reference_url in reference_urls
        if len(reference_url) > 0
    )


@pytest.mark.parametrize(
    "unseen_paper_attribute_getter", lazy_fixture(unseen_paper_attribute_getters)
)
def test_query_document_info(
    unseen_paper_attribute_getter: UnseenPaperAttributeGetter,
) -> None:
    document_info = unseen_paper_attribute_getter.get_query_document_info()

    assert isinstance(document_info, DocumentInfo)

    assert isinstance(document_info.d3_document_id, int)
    assert document_info.d3_document_id == -1

    assert isinstance(document_info.title, str)
    assert len(document_info.title) > 0

    assert isinstance(document_info.abstract, str)
    assert len(document_info.abstract) > 0

    # check that author and arxiv labels are not set
    assert document_info.author == ""
    assert document_info.arxiv_labels == []
