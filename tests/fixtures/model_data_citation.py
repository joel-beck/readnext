import pytest

from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    DocumentInfo,
)
from readnext.utils.aliases import (
    CitationFeaturesFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    InfoFrame,
    IntegerLabelsFrame,
)


@pytest.fixture(scope="session")
def citation_model_data_seen(
    citation_model_data_constructor_seen: CitationModelDataConstructor,
) -> CitationModelData:
    return CitationModelData.from_constructor(citation_model_data_constructor_seen)


@pytest.fixture(scope="session")
def citation_model_data_unseen(
    citation_model_data_constructor_unseen: CitationModelDataConstructor,
) -> CitationModelData:
    return CitationModelData.from_constructor(citation_model_data_constructor_unseen)


@pytest.fixture(scope="session")
def citation_model_data_query_document_seen(
    citation_model_data_seen: CitationModelData,
) -> DocumentInfo:
    return citation_model_data_seen.query_document


@pytest.fixture(scope="session")
def citation_model_data_query_document_unseen(
    citation_model_data_unseen: CitationModelData,
) -> DocumentInfo:
    return citation_model_data_unseen.query_document


@pytest.fixture(scope="session")
def citation_model_data_info_frame_seen(
    citation_model_data_seen: CitationModelData,
) -> InfoFrame:
    return citation_model_data_seen.info_frame


@pytest.fixture(scope="session")
def citation_model_data_info_frame_unseen(
    citation_model_data_unseen: CitationModelData,
) -> InfoFrame:
    return citation_model_data_unseen.info_frame


@pytest.fixture(scope="session")
def citation_model_data_features_frame_seen(
    citation_model_data_seen: CitationModelData,
) -> CitationFeaturesFrame:
    return citation_model_data_seen.features_frame


@pytest.fixture(scope="session")
def citation_model_data_features_frame_unseen(
    citation_model_data_unseen: CitationModelData,
) -> CitationFeaturesFrame:
    return citation_model_data_unseen.features_frame


@pytest.fixture(scope="session")
def citation_model_data_ranks_frame_seen(
    citation_model_data_seen: CitationModelData,
) -> CitationRanksFrame:
    return citation_model_data_seen.ranks_frame


@pytest.fixture(scope="session")
def citation_model_data_ranks_frame_unseen(
    citation_model_data_unseen: CitationModelData,
) -> CitationRanksFrame:
    return citation_model_data_unseen.ranks_frame


@pytest.fixture(scope="session")
def citation_model_data_points_frame_seen(
    citation_model_data_seen: CitationModelData,
) -> CitationPointsFrame:
    return citation_model_data_seen.points_frame


@pytest.fixture(scope="session")
def citation_model_data_points_frame_unseen(
    citation_model_data_unseen: CitationModelData,
) -> CitationPointsFrame:
    return citation_model_data_unseen.points_frame


@pytest.fixture(scope="session")
def citation_model_data_integer_labels_seen(
    citation_model_data_seen: CitationModelData,
) -> IntegerLabelsFrame:
    return citation_model_data_seen.integer_labels_frame


@pytest.fixture(scope="session")
def citation_model_data_integer_labels_unseen(
    citation_model_data_unseen: CitationModelData,
) -> IntegerLabelsFrame:
    return citation_model_data_unseen.integer_labels_frame
