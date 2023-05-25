import pandas as pd
import pytest

from readnext.data import (
    add_citation_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling import CitationModelDataConstructor, LanguageModelDataConstructor
from readnext.utils import ScoresFrame


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
