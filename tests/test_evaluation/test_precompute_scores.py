from readnext.modeling import DocumentScore
from readnext.modeling.document_info import DocumentInfo
from readnext.utils import ScoresFrame


def test_precompute_co_citations(co_citation_analysis_scores: ScoresFrame) -> None:
    assert isinstance(co_citation_analysis_scores, ScoresFrame)

    assert co_citation_analysis_scores.shape[1] == 1
    assert co_citation_analysis_scores.columns == ["scores"]
    assert co_citation_analysis_scores.index.name == "document_id"

    first_scores = co_citation_analysis_scores.iloc[0].item()
    assert isinstance(first_scores, list)

    first_score = first_scores[0]
    assert isinstance(first_score, DocumentScore)
    assert isinstance(first_score.document_info, DocumentInfo)

    assert isinstance(first_score.document_info.d3_document_id, int)
    assert first_score.document_info.title == ""
    assert first_score.document_info.author == ""
    assert first_score.document_info.arxiv_labels == []
    assert first_score.document_info.abstract == ""

    assert isinstance(first_score.score, int)


def test_precompute_co_references(bibliographic_coupling_scores: ScoresFrame) -> None:
    assert isinstance(bibliographic_coupling_scores, ScoresFrame)

    assert bibliographic_coupling_scores.shape[1] == 1
    assert bibliographic_coupling_scores.columns == ["scores"]
    assert bibliographic_coupling_scores.index.name == "document_id"

    first_scores = bibliographic_coupling_scores.iloc[0].item()
    assert isinstance(first_scores, list)

    first_score = first_scores[0]
    assert isinstance(first_score, DocumentScore)
    assert isinstance(first_score.document_info, DocumentInfo)

    assert isinstance(first_score.document_info.d3_document_id, int)
    assert first_score.document_info.title == ""
    assert first_score.document_info.author == ""
    assert first_score.document_info.arxiv_labels == []
    assert first_score.document_info.abstract == ""

    assert isinstance(first_score.score, int)


def test_precompute_cosine_similarities(tfidf_embeddings: ScoresFrame) -> None:
    assert isinstance(tfidf_embeddings, ScoresFrame)

    assert tfidf_embeddings.shape[1] == 1
    assert tfidf_embeddings.columns == ["scores"]
    assert tfidf_embeddings.index.name == "document_id"

    first_scores = tfidf_embeddings.iloc[0].item()
    assert isinstance(first_scores, list)

    first_score = first_scores[0]
    assert isinstance(first_score, DocumentScore)
    assert isinstance(first_score.document_info, DocumentInfo)

    assert isinstance(first_score.document_info.d3_document_id, int)
    assert first_score.document_info.title == ""
    assert first_score.document_info.author == ""
    assert first_score.document_info.arxiv_labels == []
    assert first_score.document_info.abstract == ""

    assert isinstance(first_score.score, float)

    # cosine similarity is between -1 and 1, but should be positive for all documents
    # here
    assert all(score.score >= 0 for score in first_scores)
    assert all(score.score <= 1 for score in first_scores)
