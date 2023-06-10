import pytest

from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
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
    ScoresFrame,
)


@pytest.fixture(scope="session")
def citation_model_data_constructor_seen(
    test_documents_frame: DocumentsFrame,
    seen_model_data_constructor_plugin: SeenModelDataConstructorPlugin,
    test_co_citation_analysis_scores: ScoresFrame,
    test_bibliographic_coupling_scores: ScoresFrame,
) -> CitationModelDataConstructor:
    query_d3_document_id = 13756489

    return CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
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
    query_d3_document_id = 13756489

    return CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=test_documents_frame,
        constructor_plugin=unseen_model_data_constructor_plugin,
        co_citation_analysis_scores_frame=test_co_citation_analysis_scores,
        bibliographic_coupling_scores_frame=test_bibliographic_coupling_scores,
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_query_document_seen(
    citation_model_data_constructor_seen: CitationModelDataConstructor,
) -> DocumentInfo:
    return citation_model_data_constructor_seen.query_document


@pytest.fixture(scope="session")
def citation_model_data_constructor_query_document_unseen(
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
) -> DocumentInfo:
    return citation_model_data_constructor_unseen.query_document


@pytest.fixture(scope="session")
def citation_model_data_constructor_info_frame_seen(
    citation_model_data_constructor_seen: CitationModelDataConstructor,
) -> InfoFrame:
    return citation_model_data_constructor_seen.get_info_frame()


@pytest.fixture(scope="session")
def citation_model_data_constructor_info_frame_unseen(
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
) -> InfoFrame:
    return citation_model_data_constructor_unseen.get_info_frame()


@pytest.fixture(scope="session")
def citation_model_data_constructor_features_frame_seen(
    citation_model_data_constructor_seen: CitationModelDataConstructor,
) -> CitationFeaturesFrame:
    return citation_model_data_constructor_seen.get_features_frame()


@pytest.fixture(scope="session")
def citation_model_data_constructor_features_frame_unseen(
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
) -> CitationFeaturesFrame:
    return citation_model_data_constructor_unseen.get_features_frame()


@pytest.fixture(scope="session")
def citation_model_data_constructor_ranks_frame_seen(
    citation_model_data_constructor_seen: CitationModelDataConstructor,
    citation_model_data_constructor_features_frame_seen: CitationFeaturesFrame,
) -> CitationRanksFrame:
    return citation_model_data_constructor_seen.get_ranks_frame(
        citation_model_data_constructor_features_frame_seen
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_ranks_frame_unseen(
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
    citation_model_data_constructor_features_frame_unseen: CitationFeaturesFrame,
) -> CitationRanksFrame:
    return citation_model_data_constructor_unseen.get_ranks_frame(
        citation_model_data_constructor_features_frame_unseen
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_points_frame_seen(
    citation_model_data_constructor_seen: CitationModelDataConstructor,
    citation_model_data_constructor_ranks_frame_seen: CitationRanksFrame,
) -> CitationPointsFrame:
    return citation_model_data_constructor_seen.get_points_frame(
        citation_model_data_constructor_ranks_frame_seen
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_points_frame_unseen(
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
    citation_model_data_constructor_ranks_frame_unseen: CitationRanksFrame,
) -> CitationPointsFrame:
    return citation_model_data_constructor_unseen.get_points_frame(
        citation_model_data_constructor_ranks_frame_unseen
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_integer_labels_frame_seen(
    citation_model_data_constructor_seen: CitationModelDataConstructor,
) -> IntegerLabelsFrame:
    return citation_model_data_constructor_seen.get_integer_labels_frame()


@pytest.fixture(scope="session")
def citation_model_data_constructor_integer_labels_frame_unseen(
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
) -> IntegerLabelsFrame:
    return citation_model_data_constructor_unseen.get_integer_labels_frame()
