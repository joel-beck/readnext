import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelDataConstructor,
    SeenModelDataConstructorPlugin,
    UnseenModelDataConstructorPlugin,
)
from readnext.utils.aliases import (
    CitationFeaturesFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    DocumentsFrame,
    InfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
    ScoresFrame,
)


# SECTION: Base Fixtures for Citation/Language and Seen/Unseen
@pytest.fixture(scope="session")
def citation_model_data_constructor_seen(
    test_documents_frame: DocumentsFrame,
    seen_model_data_constructor_plugin: SeenModelDataConstructorPlugin,
    test_co_citation_analysis_scores: ScoresFrame,
    test_bibliographic_coupling_scores: ScoresFrame,
) -> CitationModelDataConstructor:
    return CitationModelDataConstructor(
        d3_document_id=13756489,
        documents_frame=test_documents_frame,
        constructor_plugin=seen_model_data_constructor_plugin,
        co_citation_analysis_scores_frame=test_co_citation_analysis_scores,
        bibliographic_coupling_scores_frame=test_bibliographic_coupling_scores,
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_unseen(
    test_documents_frame: DocumentsFrame,
    unseen_model_data_constructor_plugin: UnseenModelDataConstructorPlugin,
    test_co_citation_analysis_scores: ScoresFrame,
    test_bibliographic_coupling_scores: ScoresFrame,
) -> CitationModelDataConstructor:
    return CitationModelDataConstructor(
        # TODO: Why do I have to pass the document id as input here when it is set by
        # the constructor plugin?
        d3_document_id=-1,
        documents_frame=test_documents_frame,
        constructor_plugin=unseen_model_data_constructor_plugin,
        co_citation_analysis_scores_frame=test_co_citation_analysis_scores,
        bibliographic_coupling_scores_frame=test_bibliographic_coupling_scores,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor_seen(
    test_documents_frame: DocumentsFrame,
    seen_model_data_constructor_plugin: SeenModelDataConstructorPlugin,
    test_bert_cosine_similarities: ScoresFrame,
) -> LanguageModelDataConstructor:
    return LanguageModelDataConstructor(
        d3_document_id=13756489,
        documents_frame=test_documents_frame,
        constructor_plugin=seen_model_data_constructor_plugin,
        cosine_similarity_scores_frame=test_bert_cosine_similarities,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor_unseen(
    test_documents_frame: DocumentsFrame,
    unseen_model_data_constructor_plugin: UnseenModelDataConstructorPlugin,
    test_bert_cosine_similarities: ScoresFrame,
) -> LanguageModelDataConstructor:
    return LanguageModelDataConstructor(
        d3_document_id=-1,
        documents_frame=test_documents_frame,
        constructor_plugin=unseen_model_data_constructor_plugin,
        cosine_similarity_scores_frame=test_bert_cosine_similarities,
    )


# SECTION: Model Data Constructor Attributes and Methods
citation_language_constructor_pair = [
    lazy_fixture("citation_model_data_constructor_seen"),
    lazy_fixture("language_model_data_constructor_seen"),
]

citation_seen_unseen_constructor_pair = [
    lazy_fixture("citation_model_data_constructor_seen"),
    lazy_fixture("citation_model_data_constructor_unseen"),
]

language_seen_unseen_constructor_pair = [
    lazy_fixture("language_model_data_constructor_seen"),
    lazy_fixture("language_model_data_constructor_unseen"),
]

all_constructor_fixtures = (
    citation_seen_unseen_constructor_pair + language_seen_unseen_constructor_pair
)


# query document id differs for seen and unseen papers, need two different fixtures for
# separate tests
@pytest.fixture(scope="session", params=citation_language_constructor_pair)
def model_data_constructor_query_document_seen(
    request: pytest.FixtureRequest,
) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=citation_language_constructor_pair)
def model_data_constructor_query_document_unseen(
    request: pytest.FixtureRequest,
) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=all_constructor_fixtures)
def model_data_constructor_info_frame(request: pytest.FixtureRequest) -> InfoFrame:
    return request.param.get_info_frame()


@pytest.fixture(scope="session", params=all_constructor_fixtures)
def model_data_constructor_integer_labels_frame(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.get_integer_labels_frame()


@pytest.fixture(scope="session", params=citation_seen_unseen_constructor_pair)
def citation_model_data_constructor_features_frame(
    request: pytest.FixtureRequest,
) -> CitationFeaturesFrame:
    return request.param.get_features_frame()


@pytest.fixture(scope="session", params=language_seen_unseen_constructor_pair)
def language_model_data_constructor_features_frame(
    request: pytest.FixtureRequest,
) -> LanguageFeaturesFrame:
    return request.param.get_features_frame()


@pytest.fixture(scope="session", params=citation_seen_unseen_constructor_pair)
def citation_model_data_constructor_ranks_frame(
    request: pytest.FixtureRequest,
) -> CitationRanksFrame:
    return request.param.get_ranks_frame()


@pytest.fixture(scope="session", params=citation_seen_unseen_constructor_pair)
def citation_model_data_constructor_points_frame(
    request: pytest.FixtureRequest,
) -> CitationPointsFrame:
    return request.param.get_points_frame()
