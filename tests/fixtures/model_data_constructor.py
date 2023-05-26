import pandas as pd
import pytest

from readnext.data import (
    add_citation_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelDataConstructor,
)
from readnext.utils import ScoresFrame


# SECTION: CitationModelDataConstructor
@pytest.fixture(scope="session")
def citation_model_data_constructor(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
    test_co_citation_analysis_scores_most_cited: ScoresFrame,
    test_bibliographic_coupling_scores_most_cited: ScoresFrame,
) -> CitationModelDataConstructor:
    query_d3_document_id = 13756489

    return CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=test_documents_authors_labels_citations_most_cited.pipe(
            add_citation_feature_rank_cols
        ).pipe(set_missing_publication_dates_to_max_rank),
        co_citation_analysis_scores=test_co_citation_analysis_scores_most_cited,
        bibliographic_coupling_scores=test_bibliographic_coupling_scores_most_cited,
    )


@pytest.fixture(scope="session")
def citation_model_data_constructor_query_document(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> DocumentInfo:
    return citation_model_data_constructor.query_document


@pytest.fixture(scope="session")
def citation_model_data_constructor_integer_labels(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> pd.Series:
    return citation_model_data_constructor.get_integer_labels()


@pytest.fixture(scope="session")
def citation_model_data_constructor_info_matrix(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> pd.DataFrame:
    info_matrix = citation_model_data_constructor.get_info_matrix()
    return citation_model_data_constructor.extend_info_matrix(info_matrix)


@pytest.fixture(scope="session")
def citation_model_data_constructor_feature_matrix(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> pd.DataFrame:
    return citation_model_data_constructor.get_feature_matrix()


# SECTION: LanguageModelDataConstructor
@pytest.fixture(scope="session")
def language_model_data_constructor(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
    test_bert_cosine_similarities_most_cited: ScoresFrame,
) -> LanguageModelDataConstructor:
    query_d3_document_id = 13756489

    return LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=test_documents_authors_labels_citations_most_cited,
        cosine_similarities=test_bert_cosine_similarities_most_cited,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor_query_document(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> DocumentInfo:
    return language_model_data_constructor.query_document


@pytest.fixture(scope="session")
def language_model_data_constructor_integer_labels(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> pd.Series:
    return language_model_data_constructor.get_integer_labels()


@pytest.fixture(scope="session")
def language_model_data_constructor_info_matrix(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> pd.DataFrame:
    info_matrix = language_model_data_constructor.get_info_matrix()
    return language_model_data_constructor.extend_info_matrix(info_matrix)


@pytest.fixture(scope="session")
def language_model_data_constructor_cosine_similarity_ranks(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> pd.DataFrame:
    return language_model_data_constructor.get_cosine_similarity_ranks()