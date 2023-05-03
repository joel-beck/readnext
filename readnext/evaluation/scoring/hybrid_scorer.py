from dataclasses import dataclass, field

import pandas as pd

from readnext.evaluation.metrics import AveragePrecision, CountUniqueLabels
from readnext.evaluation.scoring.model_scorer import (
    CitationModelScorer,
    FeatureWeights,
    LanguageModelScorer,
)
from readnext.modeling import CitationModelData, LanguageModelData


@dataclass(kw_only=True)
class HybridScorer:
    """
    Evaluates a hybrid recommender system by computing scores and selecting the best
    recommendations for a given query document. Stores the results for both hybrid
    component oders as well as the intermediate candidate lists.
    """

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
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
    ) -> None:
        """
        Store the intermediate candidate list, their scores and their document ids from
        a Citation -> Language hybrid recommender in instance attributes.
        """
        # compute again for each method call since a different scoring feature or number
        # of candidates may be used
        self.citation_to_language_candidates = CitationModelScorer.display_top_n(
            self.citation_model_data, feature_weights, n=n_candidates
        )

        self.citation_to_language_candidate_scores = CitationModelScorer.score_top_n(
            self.citation_model_data,
            metric,
            feature_weights=feature_weights,
            n=n_candidates,
        )

        self.citation_to_language_candidate_ids = self.citation_to_language_candidates.index

    def top_n_citation_to_language(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        """
        Select the top-n recommendations from a Citation -> Language hybrid recommender
        and set them as an instance attribute.
        """
        self.set_citation_to_language_candidates(metric, feature_weights, n_candidates)

        self.citation_to_language_recommendations = LanguageModelScorer.display_top_n(
            self.language_model_data[self.citation_to_language_candidate_ids], n=n_final
        )

    def score_citation_to_language(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        """
        Score the top-n recommendations from a Citation -> Language hybrid recommender
        and set them as an instance attribute.
        """
        self.set_citation_to_language_candidates(metric, feature_weights, n_candidates)

        self.citation_to_language_scores = LanguageModelScorer.score_top_n(
            self.language_model_data[self.citation_to_language_candidate_ids], metric, n=n_final
        )

    def set_language_to_citation_candidates(
        self, metric: AveragePrecision | CountUniqueLabels, n_candidates: int = 30
    ) -> None:
        """
        Store the intermediate candidate list, their scores and their document ids from
        a Language -> Citation hybrid recommender in instance attributes.
        """
        self.language_to_citation_candidates = LanguageModelScorer.display_top_n(
            self.language_model_data, n=n_candidates
        )

        self.language_to_citation_candidate_scores = LanguageModelScorer.score_top_n(
            self.language_model_data, metric, n=n_candidates
        )

        self.language_to_citation_candidate_ids = self.language_to_citation_candidates.index

    def top_n_language_to_citation(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        """
        Select the top-n recommendations from a Language -> Citation hybrid recommender
        and set them as an instance attribute.
        """
        self.set_language_to_citation_candidates(metric, n_candidates)

        self.language_to_citation_recommendations = CitationModelScorer.display_top_n(
            self.citation_model_data[self.language_to_citation_candidate_ids],
            feature_weights,
            n=n_final,
        )

    def score_language_to_citation(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        """
        Score the top-n recommendations from a Language -> Citation hybrid recommender
        and set them as an instance attribute.
        """
        self.set_language_to_citation_candidates(metric, n_candidates)

        self.language_to_citation_scores = CitationModelScorer.score_top_n(
            self.citation_model_data[self.language_to_citation_candidate_ids],
            metric,
            feature_weights,
            n=n_final,
        )

    def fit(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> None:
        """
        Fit a hybrid recommender model by computing and setting the top-n
        recommendations and their scores for both hybrid recommender component orders as
        instance attributes.
        """
        self.top_n_citation_to_language(metric, feature_weights, n_candidates, n_final)
        self.score_citation_to_language(metric, feature_weights, n_candidates, n_final)
        self.top_n_language_to_citation(metric, feature_weights, n_candidates, n_final)
        self.score_language_to_citation(metric, feature_weights, n_candidates, n_final)
