import pandas as pd
import pytest

from readnext.evaluation.metrics import CountCommonCitations
from readnext.evaluation.scoring import (
    find_top_n_matches_single_document,
    precompute_co_citations,
    precompute_co_references,
    precompute_cosine_similarities,
)
from readnext.modeling import DocumentScore
from readnext.modeling.document_info import DocumentInfo


@pytest.fixture
def co_citation_analysis_scores(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> pd.DataFrame:
    return precompute_co_citations(test_documents_authors_labels_citations_most_cited.head(10))


@pytest.fixture
def bibliographic_coupling_scores(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> pd.DataFrame:
    return precompute_co_references(test_documents_authors_labels_citations_most_cited.head(10))


@pytest.fixture
def tfidf_embeddings_mapping(
    test_tfidf_embeddings_mapping_most_cited: pd.DataFrame,
) -> pd.DataFrame:
    return precompute_cosine_similarities(test_tfidf_embeddings_mapping_most_cited.head(10))


def test_find_top_n_matches_single_document(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    document_ids = (
        test_documents_authors_labels_citations_most_cited["document_id"].iloc[:8].tolist()
    )
    query_document_id = document_ids[0]
    pairwise_metric = CountCommonCitations()
    n = 5

    top_n_matches = find_top_n_matches_single_document(
        test_documents_authors_labels_citations_most_cited,
        document_ids,
        query_document_id,
        pairwise_metric,
        n,
    )

    assert isinstance(top_n_matches, list)
    assert len(top_n_matches) == 5

    top_match = top_n_matches[0]
    assert isinstance(top_match, DocumentScore)
    assert isinstance(top_match.document_info, DocumentInfo)
    assert isinstance(top_match.document_info.document_id, int)
    assert top_match.document_info.title == ""
    assert top_match.document_info.author == ""
    assert top_match.document_info.arxiv_labels == []
    assert top_match.document_info.abstract == ""
    assert isinstance(top_match.score, int)

    # Check that the top match is the highest scoring match
    assert all(top_match.score >= match.score for match in top_n_matches)


def test_precompute_co_citations(co_citation_analysis_scores: pd.DataFrame) -> None:
    assert isinstance(co_citation_analysis_scores, pd.DataFrame)

    assert co_citation_analysis_scores.shape[1] == 1
    assert co_citation_analysis_scores.columns == ["scores"]
    assert co_citation_analysis_scores.index.name == "document_id"

    first_scores = co_citation_analysis_scores.iloc[0].item()
    assert isinstance(first_scores, list)

    first_score = first_scores[0]
    assert isinstance(first_score, DocumentScore)
    assert isinstance(first_score.document_info, DocumentInfo)
    assert isinstance(first_score.document_info.document_id, int)
    assert first_score.document_info.title == ""
    assert first_score.document_info.author == ""
    assert first_score.document_info.arxiv_labels == []
    assert first_score.document_info.abstract == ""

    assert isinstance(first_score.score, int)


def test_precompute_co_references(bibliographic_coupling_scores: pd.DataFrame) -> None:
    assert isinstance(bibliographic_coupling_scores, pd.DataFrame)

    assert bibliographic_coupling_scores.shape[1] == 1
    assert bibliographic_coupling_scores.columns == ["scores"]
    assert bibliographic_coupling_scores.index.name == "document_id"

    first_scores = bibliographic_coupling_scores.iloc[0].item()
    assert isinstance(first_scores, list)

    first_score = first_scores[0]
    assert isinstance(first_score, DocumentScore)
    assert isinstance(first_score.document_info, DocumentInfo)
    assert isinstance(first_score.document_info.document_id, int)
    assert first_score.document_info.title == ""
    assert first_score.document_info.author == ""
    assert first_score.document_info.arxiv_labels == []
    assert first_score.document_info.abstract == ""

    assert isinstance(first_score.score, int)


def test_precompute_cosine_similarities(tfidf_embeddings_mapping: pd.DataFrame) -> None:
    assert isinstance(tfidf_embeddings_mapping, pd.DataFrame)

    assert tfidf_embeddings_mapping.shape[1] == 1
    assert tfidf_embeddings_mapping.columns == ["scores"]
    assert tfidf_embeddings_mapping.index.name == "document_id"

    first_scores = tfidf_embeddings_mapping.iloc[0].item()
    assert isinstance(first_scores, list)

    first_score = first_scores[0]
    assert isinstance(first_score, DocumentScore)
    assert isinstance(first_score.document_info, DocumentInfo)
    assert isinstance(first_score.document_info.document_id, int)
    assert first_score.document_info.title == ""
    assert first_score.document_info.author == ""
    assert first_score.document_info.arxiv_labels == []
    assert first_score.document_info.abstract == ""

    assert isinstance(first_score.score, float)

    # cosine similarity is between -1 and 1, but should be positive for all documents
    # here
    assert all(score.score >= 0 for score in first_scores)
    assert all(score.score <= 1 for score in first_scores)
