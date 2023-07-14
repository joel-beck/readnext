import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_frame_not_equal


@pytest.mark.skip_ci
def test_different_language_models_lead_to_different_recommendations(
    tfidf_recommendations_candidates_verbose: pl.DataFrame,
    bm25_recommendations_candidates_verbose: pl.DataFrame,
    tfidf_recommendations_candidates_non_verbose: pl.DataFrame,
    bm25_recommendations_candidates_non_verbose: pl.DataFrame,
) -> None:
    assert_frame_not_equal(
        tfidf_recommendations_candidates_verbose, bm25_recommendations_candidates_verbose
    )
    assert_frame_not_equal(
        tfidf_recommendations_candidates_non_verbose, bm25_recommendations_candidates_non_verbose
    )


@pytest.mark.skip_ci
def test_verbose_parameter_does_not_affect_recommendations(
    tfidf_recommendations_candidates_verbose: pl.DataFrame,
    bm25_recommendations_candidates_verbose: pl.DataFrame,
    tfidf_recommendations_candidates_non_verbose: pl.DataFrame,
    bm25_recommendations_candidates_non_verbose: pl.DataFrame,
) -> None:
    assert_frame_equal(
        tfidf_recommendations_candidates_verbose, tfidf_recommendations_candidates_non_verbose
    )
    assert_frame_equal(
        bm25_recommendations_candidates_verbose, bm25_recommendations_candidates_non_verbose
    )
