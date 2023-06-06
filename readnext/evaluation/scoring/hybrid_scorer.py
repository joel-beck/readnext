from dataclasses import dataclass, field

import polars as pl

from readnext.config import MagicNumbers
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

    citation_model_scorer: CitationModelScorer = field(init=False)
    language_model_scorer: LanguageModelScorer = field(init=False)

    citation_to_language_candidates: pl.DataFrame = field(init=False)
    citation_to_language_candidate_ids: pl.Series = field(init=False)
    citation_to_language_candidates_score: float = field(init=False)
    citation_to_language_score: float = field(init=False)
    citation_to_language_recommendations: pl.DataFrame = field(init=False)

    language_to_citation_candidates: pl.DataFrame = field(init=False)
    language_to_citation_candidate_ids: pl.Series = field(init=False)
    language_to_citation_candidates_score: float = field(init=False)
    language_to_citation_score: float = field(init=False)
    language_to_citation_recommendations: pl.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.citation_model_scorer = CitationModelScorer(self.citation_model_data)
        self.language_model_scorer = LanguageModelScorer(self.language_model_data)

    def set_citation_to_language_candidates(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
    ) -> None:
        """
        Store the intermediate candidate list from a Citation ->
        Language hybrid recommender in instance attributes.
        """
        # compute again for each method call since a different scoring feature or number
        # of candidates may be used
        self.citation_to_language_candidates = self.citation_model_scorer.display_top_n(
            feature_weights, n=n_candidates
        )

    def set_citation_to_language_candidate_ids(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
    ) -> None:
        """
        Store the intermediate candidate document ids from a Citation ->
        Language hybrid recommender in instance attributes.
        """
        # compute again for each method call since a different scoring feature or number
        # of candidates may be used
        self.set_citation_to_language_candidates(feature_weights, n_candidates)

        self.citation_to_language_candidate_ids = self.citation_to_language_candidates[
            "candidate_d3_document_id"
        ]

    def set_citation_to_language_candidate_score(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
    ) -> None:
        """
        Store the intermediate candidate scores from a Citation -> Language hybrid
        recommender in instance attributes.
        """
        # compute again for each method call since a different scoring feature or number
        # of candidates may be used
        self.set_citation_to_language_candidate_ids(feature_weights, n_candidates)
        self.citation_to_language_candidates_score = self.citation_model_scorer.score_top_n(
            metric, feature_weights=feature_weights, n=n_candidates
        )

    def top_n_citation_to_language(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
        n_final: int = MagicNumbers.n_recommendations,
    ) -> None:
        """
        Select the top-n recommendations from a Citation -> Language hybrid recommender
        and set them as an instance attribute.
        """
        self.set_citation_to_language_candidate_ids(feature_weights, n_candidates)

        self.citation_to_language_recommendations = LanguageModelScorer.display_top_n(
            self.language_model_data[self.citation_to_language_candidate_ids.to_list()], n=n_final
        )

    def score_citation_to_language(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
        n_final: int = MagicNumbers.n_recommendations,
    ) -> None:
        """
        Score the top-n recommendations from a Citation -> Language hybrid recommender
        and set them as an instance attribute.
        """
        self.set_citation_to_language_candidate_score(metric, feature_weights, n_candidates)

        self.citation_to_language_score = LanguageModelScorer.score_top_n(
            self.language_model_data[self.citation_to_language_candidate_ids.to_list()],
            metric,
            n=n_final,
        )

    def set_language_to_citation_candidates(
        self, n_candidates: int = MagicNumbers.n_candidates
    ) -> None:
        """
        Store the intermediate candidate list from a Language -> Citation hybrid
        recommender in instance attributes.
        """
        self.language_to_citation_candidates = LanguageModelScorer.display_top_n(
            self.language_model_data, n=n_candidates
        )

    def set_language_to_citation_candidate_ids(
        self, n_candidates: int = MagicNumbers.n_candidates
    ) -> None:
        """
        Store the intermediate candidate document ids from
        a Language -> Citation hybrid recommender in instance attributes.
        """
        self.set_language_to_citation_candidates(n_candidates)

        self.language_to_citation_candidate_ids = self.language_to_citation_candidates[
            "candidate_d3_document_id"
        ]

    def set_language_to_citation_candidate_score(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        n_candidates: int = MagicNumbers.n_candidates,
    ) -> None:
        """
        Store the intermediate candidate list, their scores and their document ids from
        a Language -> Citation hybrid recommender in instance attributes.
        """
        self.set_language_to_citation_candidate_ids(n_candidates)

        self.language_to_citation_candidates_score = LanguageModelScorer.score_top_n(
            self.language_model_data, metric, n=n_candidates
        )

    def top_n_language_to_citation(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
        n_final: int = MagicNumbers.n_recommendations,
    ) -> None:
        """
        Select the top-n recommendations from a Language -> Citation hybrid recommender
        and set them as an instance attribute.
        """
        self.set_language_to_citation_candidate_ids(n_candidates)

        self.language_to_citation_recommendations = self.citation_model_scorer.display_top_n(
            # TODO: How to pass a different input to the scorer?
            self.citation_model_data[self.language_to_citation_candidate_ids.to_list()],
            feature_weights,
            n=n_final,
        )

    def score_language_to_citation(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
        n_final: int = MagicNumbers.n_recommendations,
    ) -> None:
        """
        Score the top-n recommendations from a Language -> Citation hybrid recommender
        and set them as an instance attribute.
        """
        self.set_language_to_citation_candidate_score(metric, n_candidates)

        self.language_to_citation_score = self.citation_model_scorer.score_top_n(
            self.citation_model_data[self.language_to_citation_candidate_ids.to_list()],
            metric,
            feature_weights,
            n=n_final,
        )

    def recommend(
        self,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
        n_final: int = MagicNumbers.n_recommendations,
    ) -> None:
        """
        Return the top-n recommendations from a hybrid recommender for both orders
        Language -> Citation and Citation -> Language.
        """
        self.top_n_citation_to_language(feature_weights, n_candidates, n_final)
        self.top_n_language_to_citation(feature_weights, n_candidates, n_final)

    def score(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
        n_final: int = MagicNumbers.n_recommendations,
    ) -> None:
        """
        Score the top-n recommendations from a hybrid recommender for both orders
        Language -> Citation and Citation -> Language.
        """
        self.score_citation_to_language(metric, feature_weights, n_candidates, n_final)
        self.score_language_to_citation(metric, feature_weights, n_candidates, n_final)

    def fit(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
        n_final: int = MagicNumbers.n_recommendations,
    ) -> None:
        """
        Fit a hybrid recommender model by computing the top-n recommendations and their
        scores for both orders Language -> Citation and Citation -> Language.
        """
        self.recommend(feature_weights, n_candidates, n_final)
        self.score(metric, feature_weights, n_candidates, n_final)
