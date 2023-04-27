from dataclasses import dataclass, field

import pandas as pd

from readnext.evaluation import CitationModelScorer, LanguageModelScorer, ScoringFeature
from readnext.modeling import (
    CitationModelData,
    LanguageModelData,
)


@dataclass
class HybridRecommender:
    language_model_data: LanguageModelData
    citation_model_data: CitationModelData

    citation_to_language_candidates: pd.DataFrame = field(init=False)
    citation_to_language_candidate_ids: pd.Index = field(init=False)
    citation_to_language_candidate_scores: float = field(init=False)
    language_to_citation_candidates: pd.DataFrame = field(init=False)
    language_to_citation_candidate_ids: pd.Index = field(init=False)
    language_to_citation_candidate_scores: float = field(init=False)

    def set_citation_to_language_candidates(
        self, scoring_feature: ScoringFeature = ScoringFeature.weighted, n_candidates: int = 30
    ) -> None:
        # compute again for each method call since a different scoring feature or number
        # of candidates may be used
        self.citation_to_language_candidates = CitationModelScorer.display_top_n(
            self.citation_model_data, scoring_feature, n=n_candidates
        )

        self.citation_to_language_candidate_scores = CitationModelScorer.score_top_n(
            self.citation_model_data,
            scoring_feature=ScoringFeature.weighted,
            n=n_candidates,
        )

        self.citation_to_language_candidate_ids = self.citation_to_language_candidates.index

    def select_citation_to_language(
        self,
        scoring_feature: ScoringFeature = ScoringFeature.weighted,
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> pd.DataFrame:
        self.set_citation_to_language_candidates(scoring_feature, n_candidates)

        return LanguageModelScorer.display_top_n(
            self.language_model_data[self.citation_to_language_candidate_ids], n=n_final
        )

    def score_citation_to_language(
        self,
        scoring_feature: ScoringFeature = ScoringFeature.weighted,
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> float:
        self.set_citation_to_language_candidates(scoring_feature, n_candidates)

        return LanguageModelScorer.score_top_n(
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

    def select_language_to_citation(
        self,
        scoring_feature: ScoringFeature = ScoringFeature.weighted,
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> pd.DataFrame:
        self.set_language_to_citation_candidates(n_candidates)

        return CitationModelScorer.display_top_n(
            self.citation_model_data[self.language_to_citation_candidate_ids],
            scoring_feature,
            n=n_final,
        )

    def score_language_to_citation(
        self,
        scoring_feature: ScoringFeature = ScoringFeature.weighted,
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> float:
        self.set_language_to_citation_candidates(n_candidates)

        return CitationModelScorer.score_top_n(
            self.citation_model_data[self.language_to_citation_candidate_ids],
            scoring_feature,
            n=n_final,
        )


@dataclass
class HybridScores:
    citation_to_language: float
    citation_to_language_candidates: float
    language_to_citation: float
    language_to_citation_candidates: float

    def __str__(self) -> str:
        return (
            f"Citation -> Language\n"
            "---------------------\n"
            f"Candidates Score: {self.citation_to_language_candidates:.3f}\n"
            f"Final Score: {self.citation_to_language:.3f}\n\n"
            f"Language -> Citation\n"
            "---------------------\n"
            f"Candidates Score: {self.language_to_citation_candidates:.3f}\n"
            f"Final Score: {self.language_to_citation:.3f}\n"
        )
