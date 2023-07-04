import pytest

from readnext.evaluation.scoring import HybridScore, HybridScorer
from readnext.modeling import CitationModelData, LanguageModelData


@pytest.fixture(scope="session")
def hybrid_scorer(
    citation_model_data_seen: CitationModelData, language_model_data_seen: LanguageModelData
) -> HybridScorer:
    hybrid_scorer = HybridScorer(
        language_model_name="Model",
        citation_model_data=citation_model_data_seen,
        language_model_data=language_model_data_seen,
    )
    hybrid_scorer.fit()

    return hybrid_scorer


@pytest.fixture(scope="session")
def hybrid_score(hybrid_scorer: HybridScorer) -> HybridScore:
    return HybridScore.from_scorer(hybrid_scorer)
