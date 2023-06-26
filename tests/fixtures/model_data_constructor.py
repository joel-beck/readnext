import pytest
from pytest_lazyfixture import lazy_fixture
import polars as pl

from readnext.modeling import (
    CitationModelDataConstructor,
    LanguageModelDataConstructor,
    SeenModelDataConstructorPlugin,
    UnseenModelDataConstructorPlugin,
)
from readnext.utils.aliases import (
    CandidateScoresFrame,
    CitationFeaturesFrame,
    CitationInfoFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    DocumentsFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
    LanguageInfoFrame,
    ScoresFrame,
)


# SECTION: Base Fixtures for Citation/Language and Seen/Unseen
@pytest.fixture(scope="session")
def citation_model_data_constructor_seen(
    test_documents_frame: DocumentsFrame,
    model_data_constructor_plugin_seen: SeenModelDataConstructorPlugin,
    test_co_citation_analysis_scores: ScoresFrame,
    test_bibliographic_coupling_scores: ScoresFrame,
) -> CitationModelDataConstructor:
    return CitationModelDataConstructor(
        d3_document_id=13756489,
        documents_frame=test_documents_frame,
        constructor_plugin=model_data_constructor_plugin_seen,
        co_citation_analysis_scores_frame=test_co_citation_analysis_scores,
        bibliographic_coupling_scores_frame=test_bibliographic_coupling_scores,
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_unseen(
    test_documents_frame: DocumentsFrame,
    model_data_constructor_plugin_unseen: UnseenModelDataConstructorPlugin,
    test_co_citation_analysis_scores: ScoresFrame,
    test_bibliographic_coupling_scores: ScoresFrame,
) -> CitationModelDataConstructor:
    # score frames for unseen documents only contain scores for the single unseen query
    # document and only the columns `candidate_d3_document_id` and `score`
    query_d3_document_id = test_documents_frame[0, "d3_document_id"]

    co_citation_analysis_scores_frame = test_co_citation_analysis_scores.filter(
        pl.col("query_d3_document_id") == query_d3_document_id
    ).drop("query_d3_document_id")

    bibliographic_coupling_scores_frame = test_bibliographic_coupling_scores.filter(
        pl.col("query_d3_document_id") == query_d3_document_id
    ).drop("query_d3_document_id")

    return CitationModelDataConstructor(
        # d3 document id is not used for unseen model data constructor and always set to
        # -1
        d3_document_id=-1,
        documents_frame=test_documents_frame,
        constructor_plugin=model_data_constructor_plugin_unseen,
        co_citation_analysis_scores_frame=co_citation_analysis_scores_frame,
        bibliographic_coupling_scores_frame=bibliographic_coupling_scores_frame,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor_seen(
    test_documents_frame: DocumentsFrame,
    model_data_constructor_plugin_seen: SeenModelDataConstructorPlugin,
    test_bert_cosine_similarities: ScoresFrame,
) -> LanguageModelDataConstructor:
    return LanguageModelDataConstructor(
        d3_document_id=13756489,
        documents_frame=test_documents_frame,
        constructor_plugin=model_data_constructor_plugin_seen,
        cosine_similarity_scores_frame=test_bert_cosine_similarities,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor_unseen(
    test_documents_frame: DocumentsFrame,
    model_data_constructor_plugin_unseen: UnseenModelDataConstructorPlugin,
    test_bert_cosine_similarities: ScoresFrame,
) -> LanguageModelDataConstructor:
    # score frames for unseen documents only contain scores for the single unseen query
    # document and only the columns `candidate_d3_document_id` and `score`
    query_d3_document_id = test_documents_frame[0, "d3_document_id"]

    cosine_similarity_scores_frame = test_bert_cosine_similarities.filter(
        pl.col("query_d3_document_id") == query_d3_document_id
    ).drop("query_d3_document_id")

    return LanguageModelDataConstructor(
        d3_document_id=-1,
        documents_frame=test_documents_frame,
        constructor_plugin=model_data_constructor_plugin_unseen,
        cosine_similarity_scores_frame=cosine_similarity_scores_frame,
    )


# SECTION: Model Data Constructor Attributes and Methods
citation_language_constructor_pair = [
    lazy_fixture("citation_model_data_constructor_seen"),
    lazy_fixture("language_model_data_constructor_seen"),
]

citation_constructor_pair = [
    lazy_fixture("citation_model_data_constructor_seen"),
    lazy_fixture("citation_model_data_constructor_unseen"),
]

language_constructor_pair = [
    lazy_fixture("language_model_data_constructor_seen"),
    lazy_fixture("language_model_data_constructor_unseen"),
]

seen_constructor_pair = [
    lazy_fixture("citation_model_data_constructor_seen"),
    lazy_fixture("language_model_data_constructor_seen"),
]

unseen_constructor_pair = [
    lazy_fixture("citation_model_data_constructor_unseen"),
    lazy_fixture("language_model_data_constructor_unseen"),
]

all_constructor_fixtures = citation_constructor_pair + language_constructor_pair


@pytest.fixture(scope="session", params=seen_constructor_pair)
def model_data_constructor_seen_integer_labels_frame(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.get_integer_labels_frame()


@pytest.fixture(scope="session", params=unseen_constructor_pair)
def model_data_constructor_unseen_integer_labels_frame(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.get_integer_labels_frame()


@pytest.fixture(scope="session", params=citation_constructor_pair)
def citation_model_data_constructor_info_frame(
    request: pytest.FixtureRequest,
) -> CitationInfoFrame:
    return request.param.get_info_frame()


@pytest.fixture(scope="session", params=language_constructor_pair)
def language_model_data_constructor_info_frame(
    request: pytest.FixtureRequest,
) -> LanguageInfoFrame:
    return request.param.get_info_frame()


@pytest.fixture(scope="session", params=citation_constructor_pair)
def citation_model_data_constructor_co_citation_analysis_scores(
    request: pytest.FixtureRequest,
) -> CandidateScoresFrame:
    return request.param.get_co_citation_analysis_scores()


@pytest.fixture(scope="session", params=citation_constructor_pair)
def citation_model_data_constructor_bibliographic_coupling_scores(
    request: pytest.FixtureRequest,
) -> CandidateScoresFrame:
    return request.param.get_bibliographic_coupling_scores()


@pytest.fixture(scope="session", params=citation_constructor_pair)
def citation_model_data_constructor_features_frame(
    request: pytest.FixtureRequest,
) -> CitationFeaturesFrame:
    return request.param.get_features_frame()


@pytest.fixture(scope="session", params=language_constructor_pair)
def language_model_data_constructor_features_frame(
    request: pytest.FixtureRequest,
) -> LanguageFeaturesFrame:
    return request.param.get_features_frame()


@pytest.fixture(scope="session", params=citation_constructor_pair)
def citation_model_data_constructor_ranks_frame(
    request: pytest.FixtureRequest,
    citation_model_data_constructor_features_frame: CitationFeaturesFrame,
) -> CitationRanksFrame:
    return request.param.get_ranks_frame(citation_model_data_constructor_features_frame)


@pytest.fixture(scope="session", params=citation_constructor_pair)
def citation_model_data_constructor_points_frame(
    request: pytest.FixtureRequest,
    citation_model_data_constructor_ranks_frame: CitationRanksFrame,
) -> CitationPointsFrame:
    return request.param.get_points_frame(citation_model_data_constructor_ranks_frame)
