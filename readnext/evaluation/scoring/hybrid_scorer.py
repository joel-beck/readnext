from dataclasses import dataclass, field

import pandas as pd

from readnext.evaluation.scoring.model_scorer import (
    CitationModelScorer,
    FeatureWeights,
    LanguageModelScorer,
)
from readnext.modeling import CitationModelData, LanguageModelData


@dataclass(kw_only=True)
class HybridScorer:
    language_model_name: str
    language_model_data: LanguageModelData
    citation_model_data: CitationModelData

    citation_to_language_candidates: pd.DataFrame = field(init=False)
    citation_to_language_candidate_ids: pd.Index = field(init=False)
    citation_to_language_candidate_scores: float = field(init=False)
    citation_to_language_scores: float = field(init=False)
    citation_to_language_recommendations: pd.DataFrame = field(init=False)

    language_to_citation_candidates: pd.DataFrame = field(init=False)
    language_to_citation_candidate_ids: pd.Index = field(init=False)
    language_to_citation_candidate_scores: float = field(init=False)
    language_to_citation_scores: float = field(init=False)
    language_to_citation_recommendations: pd.DataFrame = field(init=False)

    def set_citation_to_language_candidates(
        self, feature_weights: FeatureWeights = FeatureWeights(), n_candidates: int = 30
    ) -> None:
        # compute again for each method call since a different scoring feature or number
        # of candidates may be used
        self.citation_to_language_candidates = CitationModelScorer.display_top_n(
            self.citation_model_data, feature_weights, n=n_candidates
        )

        self.citation_to_language_candidate_scores = CitationModelScorer.score_top_n(
            self.citation_model_data,
            feature_weights=feature_weights,
            n=n_candidates,
        )

        self.citation_to_language_candidate_ids = self.citation_to_language_candidates.index

    def top_n_citation_to_language(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        self.set_citation_to_language_candidates(feature_weights, n_candidates)

        self.citation_to_language_recommendations = LanguageModelScorer.display_top_n(
            self.language_model_data[self.citation_to_language_candidate_ids], n=n_final
        )

    def score_citation_to_language(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        self.set_citation_to_language_candidates(feature_weights, n_candidates)

        self.citation_to_language_scores = LanguageModelScorer.score_top_n(
            self.language_model_data[self.citation_to_language_candidate_ids], n=n_final
        )

    def set_language_to_citation_candidates(self, n_candidates: int = 30) -> None:
        self.language_to_citation_candidates = LanguageModelScorer.display_top_n(
            self.language_model_data, n=n_candidates
        )

        self.language_to_citation_candidate_scores = LanguageModelScorer.score_top_n(
            self.language_model_data, n=n_candidates
        )

        self.language_to_citation_candidate_ids = self.language_to_citation_candidates.index

    def top_n_language_to_citation(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        self.set_language_to_citation_candidates(n_candidates)

        self.language_to_citation_recommendations = CitationModelScorer.display_top_n(
            self.citation_model_data[self.language_to_citation_candidate_ids],
            feature_weights,
            n=n_final,
        )

    def score_language_to_citation(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        self.set_language_to_citation_candidates(n_candidates)

        self.language_to_citation_scores = CitationModelScorer.score_top_n(
            self.citation_model_data[self.language_to_citation_candidate_ids],
            feature_weights,
            n=n_final,
        )

    def fit(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        self.top_n_citation_to_language(feature_weights, n_candidates, n_final)
        self.score_citation_to_language(feature_weights, n_candidates, n_final)
        self.top_n_language_to_citation(feature_weights, n_candidates, n_final)
        self.score_language_to_citation(feature_weights, n_candidates, n_final)
