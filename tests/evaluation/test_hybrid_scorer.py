import polars as pl
import pytest

from readnext.evaluation.scoring import CitationModelScorer, HybridScorer, LanguageModelScorer
from readnext.modeling import (
    CitationModelData,
    LanguageModelData,
)
from readnext.modeling.document_info import DocumentInfo


def test_basic_attributes(hybrid_scorer: HybridScorer) -> None:
    assert isinstance(hybrid_scorer, HybridScorer)

    assert isinstance(hybrid_scorer.language_model_name, str)
    assert isinstance(hybrid_scorer.citation_model_data, CitationModelData)
    assert isinstance(hybrid_scorer.language_model_data, LanguageModelData)

    assert isinstance(hybrid_scorer.citation_model_scorer_candidates, CitationModelScorer)
    assert isinstance(hybrid_scorer.language_model_scorer_candidates, LanguageModelScorer)


candidate_ids_attributes = [
    "citation_to_language_candidate_ids",
    "language_to_citation_candidate_ids",
]


@pytest.mark.parametrize("candidate_ids_attribute", candidate_ids_attributes)
def test_candidate_ids(hybrid_scorer: HybridScorer, candidate_ids_attribute: str) -> None:
    candidate_ids = getattr(hybrid_scorer, candidate_ids_attribute)
    assert isinstance(candidate_ids, list)
    assert all(isinstance(x, int) for x in candidate_ids)


score_attributes = [
    "citation_to_language_candidates_score",
    "citation_to_language_score",
    "language_to_citation_candidates_score",
    "language_to_citation_score",
]


@pytest.mark.parametrize("score_attribute", score_attributes)
def test_scores(hybrid_scorer: HybridScorer, score_attribute: str) -> None:
    score = getattr(hybrid_scorer, score_attribute)
    assert isinstance(score, float)
    assert score > 0
    assert score <= 1


citation_frame_attributes = ["citation_to_language_candidates", "language_to_citation"]


@pytest.mark.parametrize("citation_frame_attribute", citation_frame_attributes)
def test_citation_frames(hybrid_scorer: HybridScorer, citation_frame_attribute: str) -> None:
    citation_frame = getattr(hybrid_scorer, citation_frame_attribute)
    assert isinstance(citation_frame, pl.DataFrame)
    assert citation_frame.width == 18
    assert citation_frame["weighted_points"].is_sorted(descending=True)

    assert isinstance(hybrid_scorer.citation_to_language_recommendations, pl.DataFrame)
    assert hybrid_scorer.citation_to_language_recommendations.width == 9
    assert hybrid_scorer.citation_to_language_recommendations["cosine_similarity"].is_sorted(
        descending=True
    )


language_frame_attributes = ["language_to_citation_candidates", "citation_to_language"]


@pytest.mark.parametrize("language_frame_attribute", language_frame_attributes)
def test_language_frames(hybrid_scorer: HybridScorer, language_frame_attribute: str) -> None:
    language_frame = getattr(hybrid_scorer, language_frame_attribute)
    assert isinstance(language_frame, pl.DataFrame)
    assert language_frame.width == 9
    assert language_frame["cosine_similarity"].is_sorted(descending=True)


def test_kw_only_initialization_hybrid_scorer() -> None:
    with pytest.raises(TypeError):
        HybridScorer(
            "TF-IDF",  # type: ignore
            LanguageModelData(
                query_document=DocumentInfo(
                    d3_document_id=-1,
                    title="Title",
                    author="Author",
                    publication_date="2000-01-01",
                    semanticscholar_url="",
                    arxiv_url="",
                    abstract="Abstract",
                    arxiv_labels=[],
                ),
                info_frame=pl.DataFrame(),
                integer_labels_frame=pl.DataFrame(),
                features_frame=pl.DataFrame(),
            ),
            CitationModelData(
                query_document=DocumentInfo(
                    d3_document_id=-1,
                    title="Title",
                    author="Author",
                    publication_date="2000-01-01",
                    abstract="Abstract",
                    arxiv_labels=[],
                ),
                info_frame=pl.DataFrame(),
                features_frame=pl.DataFrame(),
                integer_labels_frame=pl.DataFrame(),
                ranks_frame=pl.DataFrame(),
                points_frame=pl.DataFrame(),
            ),
        )
