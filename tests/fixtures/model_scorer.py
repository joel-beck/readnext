import pytest

from readnext.evaluation.scoring import CitationModelScorer, LanguageModelScorer
from readnext.modeling import CitationModelData, LanguageModelData


@pytest.fixture(scope="session")
def citation_model_scorer(citation_model_data_seen: CitationModelData) -> CitationModelScorer:
    return CitationModelScorer(citation_model_data_seen)


@pytest.fixture(scope="session")
def language_model_scorer(language_model_data_seen: LanguageModelData) -> LanguageModelScorer:
    return LanguageModelScorer(language_model_data_seen)
