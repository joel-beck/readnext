import polars as pl
import pytest

from readnext import FeatureWeights, LanguageModelChoice, readnext


@pytest.fixture(scope="session")
def tfidf_recommendations_candidates_verbose() -> pl.DataFrame:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    recommendations = readnext(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
    )

    return recommendations.recommendations.language_to_citation_candidates


@pytest.fixture(scope="session")
def bm25_recommendations_candidates_verbose() -> pl.DataFrame:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    recommendations = readnext(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.BM25,
        feature_weights=FeatureWeights(),
    )

    return recommendations.recommendations.language_to_citation_candidates


@pytest.fixture(scope="session")
def tfidf_recommendations_candidates_non_verbose() -> pl.DataFrame:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    recommendations = readnext(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        _verbose=False,
    )

    return recommendations.recommendations.language_to_citation_candidates


@pytest.fixture(scope="session")
def bm25_recommendations_candidates_non_verbose() -> pl.DataFrame:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    recommendations = readnext(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.BM25,
        feature_weights=FeatureWeights(),
        _verbose=False,
    )

    return recommendations.recommendations.language_to_citation_candidates
