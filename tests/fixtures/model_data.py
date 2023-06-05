import polars as pl
import pytest

from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelData,
    LanguageModelDataConstructor,
)


# SECTION: CitationModelData
@pytest.fixture(scope="session")
def citation_model_data(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> CitationModelData:
    return CitationModelData.from_constructor(citation_model_data_constructor)


@pytest.fixture(scope="session")
def citation_model_data_query_document(
    citation_model_data: CitationModelData,
) -> DocumentInfo:
    return citation_model_data.query_document


@pytest.fixture(scope="session")
def citation_model_data_integer_labels(
    citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return citation_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def citation_model_data_info_matrix(
    citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return citation_model_data.info_frame


@pytest.fixture(scope="session")
def citation_model_data_feature_matrix(
    citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return citation_model_data.features_frame


# SECTION: LanguageModelData
@pytest.fixture(scope="session")
def language_model_data(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> LanguageModelData:
    return LanguageModelData.from_constructor(language_model_data_constructor)


@pytest.fixture(scope="session")
def language_model_data_query_document(
    language_model_data: LanguageModelData,
) -> DocumentInfo:
    return language_model_data.query_document


@pytest.fixture(scope="session")
def language_model_data_integer_labels(
    language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return language_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def language_model_data_info_matrix(
    language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return language_model_data.info_frame


@pytest.fixture(scope="session")
def language_model_data_cosine_similarity_ranks(
    language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return language_model_data.cosine_similarity_ranks
