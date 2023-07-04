import polars as pl
import pytest

from readnext.evaluation.scoring import HybridScore

score_attributes = [
    "citation_to_language_candidates",
    "citation_to_language",
    "language_to_citation_candidates",
    "language_to_citation",
]


@pytest.mark.parametrize("score_attribute", score_attributes)
def test_hybrid_score_attributes(hybrid_score: HybridScore, score_attribute: str) -> None:
    assert isinstance(hybrid_score, HybridScore)
    assert isinstance(hybrid_score.language_model_name, str)

    score = getattr(hybrid_score, score_attribute)
    assert isinstance(score, float)
    assert score > 0
    assert score <= 1


def test_to_frame(hybrid_score: HybridScore) -> None:
    hybrid_score_frame = hybrid_score.to_frame()

    assert isinstance(hybrid_score_frame, pl.DataFrame)
    # check that dataframe has only a single row
    assert hybrid_score_frame.height == 1
    assert hybrid_score_frame.width == 5
