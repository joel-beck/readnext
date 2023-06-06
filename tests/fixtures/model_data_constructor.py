import polars as pl
import pytest

from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelDataConstructor,
)
from readnext.utils import DocumentsFrame, ScoresFrame


# SECTION: CitationModelDataConstructor
@pytest.fixture(scope="session")
def citation_model_data_constructor(
    test_documents_frame: DocumentsFrame,
    test_co_citation_analysis_scores: ScoresFrame,
    test_bibliographic_coupling_scores: ScoresFrame,
) -> CitationModelDataConstructor:
    query_d3_document_id = 13756489

    return CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=test_documents_frame,
        co_citation_analysis_scores_frame=test_co_citation_analysis_scores,
        bibliographic_coupling_scores_frame=test_bibliographic_coupling_scores,
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_query_document(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> DocumentInfo:
    return citation_model_data_constructor.query_document


@pytest.fixture(scope="session")
def citation_model_data_constructor_integer_labels(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> pl.DataFrame:
    return citation_model_data_constructor.get_integer_labels_frame()


@pytest.fixture(scope="session")
def citation_model_data_constructor_info_matrix(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> pl.DataFrame:
    info_matrix = citation_model_data_constructor.get_info_frame()
    return citation_model_data_constructor.add_scores_to_info_matrix(info_matrix)


@pytest.fixture(scope="session")
def citation_model_data_constructor_feature_matrix(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> pl.DataFrame:
    return citation_model_data_constructor.get_features_frame()


# SECTION: LanguageModelDataConstructor
@pytest.fixture(scope="session")
def language_model_data_constructor(
    test_documents_authors_labels_citations: pl.DataFrame,
    test_bert_cosine_similarities: ScoresFrame,
) -> LanguageModelDataConstructor:
    query_d3_document_id = 13756489

    return LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=test_documents_authors_labels_citations,
        cosine_similarity_scores_frame=test_bert_cosine_similarities,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor_query_document(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> DocumentInfo:
    return language_model_data_constructor.query_document


@pytest.fixture(scope="session")
def language_model_data_constructor_integer_labels(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> pl.DataFrame:
    return language_model_data_constructor.get_integer_labels_frame()


@pytest.fixture(scope="session")
def language_model_data_constructor_info_matrix(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> pl.DataFrame:
    info_matrix = language_model_data_constructor.get_info_frame()
    return language_model_data_constructor.add_scores_to_info_matrix(info_matrix)


@pytest.fixture(scope="session")
def language_model_data_constructor_cosine_similarity_ranks(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> pl.DataFrame:
    return language_model_data_constructor.get_cosine_similarity_ranks_frame()
