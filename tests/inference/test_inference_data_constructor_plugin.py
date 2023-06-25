import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.data.semanticscholar import SemanticscholarRequest, SemanticScholarResponse
from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, InferenceDataInputConverter
from readnext.inference.constructor_plugin import InferenceDataConstructorPlugin
from readnext.inference.constructor_plugin_seen import SeenInferenceDataConstructorPlugin
from readnext.inference.constructor_plugin_unseen import UnseenInferenceDataConstructorPlugin
from readnext.modeling.constructor_plugin import (
    SeenModelDataConstructorPlugin,
    UnseenModelDataConstructorPlugin,
)
from readnext.modeling.language_models import LanguageModelChoice

inference_data_constructor_plugins_seen = [
    lazy_fixture("inference_data_constructor_plugin_seen_from_semanticscholar_id"),
    lazy_fixture("inference_data_constructor_plugin_seen_from_semanticscholar_id"),
    lazy_fixture("inference_data_constructor_plugin_seen_from_arxiv_id"),
    lazy_fixture("inference_data_constructor_plugin_seen_from_arxiv_url"),
]

inference_data_constructor_plugins_unseen = [
    lazy_fixture("inference_data_constructor_plugin_unseen_from_semanticscholar_id"),
    lazy_fixture("inference_data_constructor_plugin_unseen_from_semanticscholar_id"),
    lazy_fixture("inference_data_constructor_plugin_unseen_from_arxiv_id"),
    lazy_fixture("inference_data_constructor_plugin_unseen_from_arxiv_url"),
]

inference_data_constructor_plugins = (
    inference_data_constructor_plugins_seen + inference_data_constructor_plugins_unseen
)


# SECTION: InferenceDataConstructorPlugin
@pytest.mark.parametrize("constructor_plugins", inference_data_constructor_plugins)
def test_attributes_are_created_correctly(
    constructor_plugins: InferenceDataConstructorPlugin,
) -> None:
    assert isinstance(constructor_plugins.identifier, DocumentIdentifier)
    assert isinstance(constructor_plugins.language_model_choice, LanguageModelChoice)
    assert isinstance(constructor_plugins.feature_weights, FeatureWeights)
    assert isinstance(constructor_plugins.documents_frame, pl.DataFrame)


# SECTION: SeenInferenceDataConstructorPlugin
def test_passing_inputs_to_input_converter_works_for_semanticscholar_id(
    inference_data_constructor_plugin_seen_from_semanticscholar_id: SeenInferenceDataConstructorPlugin,  # noqa: E501
) -> None:
    assert (
        inference_data_constructor_plugin_seen_from_semanticscholar_id.semanticscholar_id
        is not None
    )
    assert isinstance(
        inference_data_constructor_plugin_seen_from_semanticscholar_id.semanticscholar_id, str
    )
    assert (
        inference_data_constructor_plugin_seen_from_semanticscholar_id.semanticscholar_url is None
    )
    assert inference_data_constructor_plugin_seen_from_semanticscholar_id.arxiv_id is None
    assert inference_data_constructor_plugin_seen_from_semanticscholar_id.arxiv_url is None


def test_passing_inputs_to_input_converter_works_for_semanticscholar_url(
    inference_data_constructor_plugin_seen_from_semanticscholar_url: SeenInferenceDataConstructorPlugin,  # noqa: E501
) -> None:
    assert (
        inference_data_constructor_plugin_seen_from_semanticscholar_url.semanticscholar_id is None
    )
    assert (
        inference_data_constructor_plugin_seen_from_semanticscholar_url.semanticscholar_url
        is not None
    )
    assert isinstance(
        inference_data_constructor_plugin_seen_from_semanticscholar_url.semanticscholar_url, str
    )
    assert inference_data_constructor_plugin_seen_from_semanticscholar_url.arxiv_id is None
    assert inference_data_constructor_plugin_seen_from_semanticscholar_url.arxiv_url is None


def test_passing_inputs_to_input_converter_works_for_arxiv_id(
    inference_data_constructor_plugin_seen_from_arxiv_id: SeenInferenceDataConstructorPlugin,
) -> None:
    assert inference_data_constructor_plugin_seen_from_arxiv_id.semanticscholar_id is None
    assert inference_data_constructor_plugin_seen_from_arxiv_id.semanticscholar_url is None
    assert inference_data_constructor_plugin_seen_from_arxiv_id.arxiv_id is not None
    assert isinstance(inference_data_constructor_plugin_seen_from_arxiv_id.arxiv_id, str)
    assert inference_data_constructor_plugin_seen_from_arxiv_id.arxiv_url is None


def test_passing_inputs_to_input_converter_works_for_arxiv_url(
    inference_data_constructor_plugin_seen_from_arxiv_url: SeenInferenceDataConstructorPlugin,
) -> None:
    assert inference_data_constructor_plugin_seen_from_arxiv_url.semanticscholar_id is None
    assert inference_data_constructor_plugin_seen_from_arxiv_url.semanticscholar_url is None
    assert inference_data_constructor_plugin_seen_from_arxiv_url.arxiv_id is None
    assert inference_data_constructor_plugin_seen_from_arxiv_url.arxiv_url is not None
    assert isinstance(inference_data_constructor_plugin_seen_from_arxiv_url.arxiv_url, str)


@pytest.mark.parametrize("constructor_plugin_seen", inference_data_constructor_plugins_seen)
def test_constructor_plugin_seen_attributes_are_created_correctly(
    constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> None:
    assert isinstance(constructor_plugin_seen.input_converter, InferenceDataInputConverter)
    assert isinstance(
        constructor_plugin_seen.model_data_constructor_plugin, SeenModelDataConstructorPlugin
    )


def test_kw_only_initialization_constructor_plugin_seen() -> None:
    with pytest.raises(TypeError):
        SeenInferenceDataConstructorPlugin(
            None,  # type: ignore
            None,
            None,
            None,
            LanguageModelChoice.TFIDF,
            FeatureWeights(),
            pl.DataFrame(),
        )


# SECTION: UnseenInferenceDataConstructorPlugin
@pytest.mark.parametrize("constructor_plugin_unseen", inference_data_constructor_plugins_unseen)
def test_constructor_plugin_unseen_attributes_are_created_correctly(
    constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> None:
    assert isinstance(constructor_plugin_unseen.semanticscholar_request, SemanticscholarRequest)

    assert isinstance(constructor_plugin_unseen.response, SemanticScholarResponse)
    assert isinstance(
        constructor_plugin_unseen.model_data_constructor_plugin, UnseenModelDataConstructorPlugin
    )


@pytest.mark.parametrize("constructor_plugin_unseen", inference_data_constructor_plugins_unseen)
def test_send_semanticscholar_request(
    constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> None:
    constructor_plugin_unseen.send_semanticscholar_request()

    assert isinstance(constructor_plugin_unseen.response, SemanticScholarResponse)

    # defaults are empty strings instead of None => check for length > 0 for set
    # attributes
    assert isinstance(constructor_plugin_unseen.response.semanticscholar_id, str)
    assert len(constructor_plugin_unseen.response.semanticscholar_id) > 0

    assert isinstance(constructor_plugin_unseen.response.arxiv_id, str)

    assert isinstance(constructor_plugin_unseen.response.title, str)
    assert len(constructor_plugin_unseen.response.title) > 0

    assert isinstance(constructor_plugin_unseen.response.abstract, str)
    assert len(constructor_plugin_unseen.response.abstract) > 0

    assert isinstance(constructor_plugin_unseen.response.citations, list)
    assert all(
        isinstance(citation, dict) for citation in constructor_plugin_unseen.response.citations
    )
    # if paperId or title is set, it is a string with length > 0
    assert all(
        (isinstance(citation["paperId"], str) and len(citation["paperId"]) > 0)
        or citation["paperId"] is None
        for citation in constructor_plugin_unseen.response.citations
    )
    assert all(
        (isinstance(citation["title"], str) and len(citation["title"]) > 0)
        or citation["title"] is None
        for citation in constructor_plugin_unseen.response.citations
    )

    assert constructor_plugin_unseen.response.references is not None
    assert isinstance(constructor_plugin_unseen.response.references, list)
    assert all(
        isinstance(reference, dict) for reference in constructor_plugin_unseen.response.references
    )
    assert all(
        (isinstance(citation["paperId"], str) and len(citation["paperId"]) > 0)
        or citation["paperId"] is None
        for citation in constructor_plugin_unseen.response.references
    )
    assert all(
        (isinstance(citation["title"], str) and len(citation["title"]) > 0)
        or citation["title"] is None
        for citation in constructor_plugin_unseen.response.references
    )


@pytest.mark.parametrize("constructor_plugin_unseen", inference_data_constructor_plugins_unseen)
def test_query_citation_urls(
    constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> None:
    citation_urls = constructor_plugin_unseen.get_query_citation_urls()

    assert isinstance(citation_urls, list)
    assert all(isinstance(citation_url, str) for citation_url in citation_urls)

    # check that all set citation urls are semanticscholar urls
    assert all(
        citation_url.startswith("https://www.semanticscholar.org/paper/")
        for citation_url in citation_urls
        if len(citation_url) > 0
    )


@pytest.mark.parametrize("constructor_plugin_unseen", inference_data_constructor_plugins_unseen)
def test_query_reference_urls(
    constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> None:
    reference_urls = constructor_plugin_unseen.get_query_reference_urls()

    assert isinstance(reference_urls, list)
    assert all(isinstance(reference_url, str) for reference_url in reference_urls)

    # check that all set reference urls are semanticscholar urls
    assert all(
        reference_url.startswith("https://www.semanticscholar.org/paper/")
        for reference_url in reference_urls
        if len(reference_url) > 0
    )


def test_kw_only_initialization_constructor_plugin_unseen() -> None:
    with pytest.raises(TypeError):
        UnseenInferenceDataConstructorPlugin(
            None,  # type: ignore
            None,
            None,
            None,
            LanguageModelChoice.TFIDF,
            FeatureWeights(),
            pl.DataFrame(),
        )
