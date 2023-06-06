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

    citation_model_scorer_candidates: CitationModelScorer = field(init=False)
    language_model_scorer_candidates: LanguageModelScorer = field(init=False)

    citation_to_language_candidates: pl.DataFrame = field(init=False)
    citation_to_language_candidate_ids: list[int] = field(init=False)
    citation_to_language_candidates_score: float = field(init=False)
    citation_to_language_score: float = field(init=False)
    citation_to_language_recommendations: pl.DataFrame = field(init=False)

    language_to_citation_candidates: pl.DataFrame = field(init=False)
    language_to_citation_candidate_ids: list[int] = field(init=False)
    language_to_citation_candidates_score: float = field(init=False)
    language_to_citation_score: float = field(init=False)
    language_to_citation_recommendations: pl.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        # compute scores to create the candidate lists
        self.citation_model_scorer_candidates = CitationModelScorer(self.citation_model_data)
        self.language_model_scorer_candidates = LanguageModelScorer(self.language_model_data)

    def get_citation_to_language_candidates(
        self, feature_weights: FeatureWeights, n_candidates: int
    ) -> pl.DataFrame:
        """
        Store the intermediate candidate list from a Citation -> Language hybrid
        recommender in instance attributes.
        """
        return self.citation_model_scorer_candidates.display_top_n(feature_weights, n_candidates)

    @staticmethod
    def get_citation_to_language_candidate_ids(
        citation_to_language_candidates: pl.DataFrame,
    ) -> list[int]:
        """
        Store the intermediate candidate document ids from a Citation -> Language hybrid
        recommender in instance attributes.
        """
        return citation_to_language_candidates["candidate_d3_document_id"].to_list()

    def get_citation_to_language_candidate_score(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights,
        n_candidates: int,
    ) -> int | float:
        """
        Store the intermediate candidate scores from a Citation -> Language hybrid
        recommender in instance attributes.
        """
        return self.citation_model_scorer_candidates.score_top_n(
            metric, feature_weights, n_candidates
        )

    def get_language_model_scorer_final(
        self, citation_to_language_candidate_ids: list[int]
    ) -> LanguageModelScorer:
        """
        Computes scores for the final ranking based on the Citation -> language
        candidates.
        """
        return LanguageModelScorer(self.language_model_data[citation_to_language_candidate_ids])

    def top_n_citation_to_language(
        self,
        language_model_scorer_final: LanguageModelScorer,
        feature_weights: FeatureWeights,
        n_final: int,
    ) -> pl.DataFrame:
        """
        Select the top-n recommendations from a Citation -> Language hybrid recommender
        and set them as an instance attribute.
        """
        return language_model_scorer_final.display_top_n(feature_weights, n_final)

    def score_citation_to_language(
        self,
        language_model_scorer_final: LanguageModelScorer,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights,
        n_final: int,
    ) -> int | float:
        """
        Score the top-n recommendations from a Citation -> Language hybrid recommender
        and set them as an instance attribute.
        """
        return language_model_scorer_final.score_top_n(metric, feature_weights, n_final)

    def set_citation_to_language_attributes(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights,
        n_candidates: int,
        n_final: int,
    ) -> None:
        self.citation_to_language_candidates = self.get_citation_to_language_candidates(
            feature_weights, n_candidates
        )
        self.citation_to_language_candidate_ids = self.get_citation_to_language_candidate_ids(
            self.citation_to_language_candidates
        )
        self.citation_to_language_candidates_score = self.get_citation_to_language_candidate_score(
            metric, feature_weights, n_candidates
        )

        language_model_scorer_final = self.get_language_model_scorer_final(
            self.citation_to_language_candidate_ids
        )
        self.citation_to_language_score = self.score_citation_to_language(
            language_model_scorer_final, metric, feature_weights, n_final
        )
        self.citation_to_language_recommendations = self.top_n_citation_to_language(
            language_model_scorer_final, feature_weights, n_final
        )

    def get_language_to_citation_candidates(
        self, feature_weights: FeatureWeights, n_candidates: int
    ) -> pl.DataFrame:
        """
        Store the intermediate candidate list from a Language -> Citation hybrid
        recommender in instance attributes.
        """
        return self.language_model_scorer_candidates.display_top_n(feature_weights, n_candidates)

    def get_language_to_citation_candidate_ids(
        self, language_to_citation_candidates: pl.DataFrame
    ) -> list[int]:
        """
        Store the intermediate candidate document ids from a Language -> Citation hybrid
        recommender in instance attributes.
        """
        return language_to_citation_candidates["candidate_d3_document_id"].to_list()

    def get_language_to_citation_candidate_score(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights,
        n_candidates: int,
    ) -> int | float:
        """
        Store the intermediate candidate scores from a Language -> Citation hybrid
        recommender in instance attributes.
        """
        return self.language_model_scorer_candidates.score_top_n(
            metric, feature_weights, n_candidates
        )

    def get_citation_model_scorer_final(
        self, language_to_citation_candidate_ids: list[int]
    ) -> CitationModelScorer:
        """
        Computes scores for the final ranking based on the Language -> Citation
        candidates.
        """
        return CitationModelScorer(self.citation_model_data[language_to_citation_candidate_ids])

    def top_n_language_to_citation(
        self,
        citation_model_scorer_final: CitationModelScorer,
        feature_weights: FeatureWeights,
        n_final: int,
    ) -> pl.DataFrame:
        """
        Select the top-n recommendations from a Language -> Citation hybrid recommender
        and set them as an instance attribute.
        """
        return citation_model_scorer_final.display_top_n(feature_weights, n_final)

    def score_language_to_citation(
        self,
        citation_model_scorer_final: CitationModelScorer,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights,
        n_final: int,
    ) -> int | float:
        """
        Score the top-n recommendations from a Language -> Citation hybrid recommender
        and set them as an instance attribute.
        """
        return citation_model_scorer_final.score_top_n(metric, feature_weights, n_final)

    def set_language_to_citation_attributes(
        self,
        metric: AveragePrecision | CountUniqueLabels,
        feature_weights: FeatureWeights,
        n_candidates: int,
        n_final: int,
    ) -> None:
        self.language_to_citation_candidates = self.get_language_to_citation_candidates(
            feature_weights, n_candidates
        )
        self.language_to_citation_candidate_ids = self.get_language_to_citation_candidate_ids(
            self.language_to_citation_candidates
        )
        self.language_to_citation_candidates_score = self.get_language_to_citation_candidate_score(
            metric, feature_weights, n_candidates
        )

        citation_model_scorer_final = self.get_citation_model_scorer_final(
            self.language_to_citation_candidate_ids
        )
        self.language_to_citation_score = self.score_language_to_citation(
            citation_model_scorer_final, metric, feature_weights, n_final
        )
        self.language_to_citation_recommendations = self.top_n_language_to_citation(
            citation_model_scorer_final, feature_weights, n_final
        )

    def fit(
        self,
        metric: AveragePrecision | CountUniqueLabels = AveragePrecision(),
        feature_weights: FeatureWeights = FeatureWeights(),
        n_candidates: int = MagicNumbers.n_candidates,
        n_final: int = MagicNumbers.n_recommendations,
    ) -> None:
        """
        Fit a hybrid recommender model by computing the top-n recommendations and their
        scores for both orders Language -> Citation and Citation -> Language.
        """
        self.set_citation_to_language_attributes(metric, feature_weights, n_candidates, n_final)
        self.set_language_to_citation_attributes(metric, feature_weights, n_candidates, n_final)
