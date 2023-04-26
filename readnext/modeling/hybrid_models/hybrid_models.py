from dataclasses import dataclass, field

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation import CitationModelScorer, LanguageModelScorer, ScoringFeature
from readnext.modeling import (
    CitationModelData,
    CitationModelDataFromId,
    LanguageModelData,
    LanguageModelDataFromId,
)


@dataclass
class HybridRecommender:
    language_model_data: LanguageModelData
    citation_model_data: CitationModelData

    candidates_citation_to_language: pd.DataFrame | None = field(init=False, default=None)
    candidate_ids_citation_to_language: pd.Index | None = field(init=False, default=None)
    candidate_scores_citation_to_language: float | None = field(init=False, default=None)
    candidates_language_to_citation: pd.DataFrame | None = field(init=False, default=None)
    candidate_ids_language_to_citation: pd.Index | None = field(init=False, default=None)
    candidate_scores_language_to_citation: float | None = field(init=False, default=None)

    def set_candidates_citation_to_language(
        self, scoring_feature: ScoringFeature = ScoringFeature.weighted, n_candidates: int = 30
    ) -> None:
        # compute again for each method call since a different scoring feature or number
        # of candidates may be used
        self.candidates_citation_to_language = CitationModelScorer.display_top_n(
            self.citation_model_data, scoring_feature, n=n_candidates
        )

        self.candidate_scores_citation_to_language = CitationModelScorer.score_top_n(
            self.citation_model_data,
            scoring_feature=ScoringFeature.weighted,
            n=n_candidates,
        )

        self.candidate_ids_citation_to_language = self.candidates_citation_to_language.index

    def select_citation_to_language(
        self,
        scoring_feature: ScoringFeature = ScoringFeature.weighted,
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> pd.DataFrame:
        self.set_candidates_citation_to_language(scoring_feature, n_candidates)

        return LanguageModelScorer.display_top_n(
            self.language_model_data[self.candidate_ids_citation_to_language], n=n_final
        )

    def score_citation_to_language(
        self,
        scoring_feature: ScoringFeature = ScoringFeature.weighted,
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> float:
        self.set_candidates_citation_to_language(scoring_feature, n_candidates)

        return LanguageModelScorer.score_top_n(
            self.language_model_data[self.candidate_ids_citation_to_language], n=n_final
        )

    def set_candidates_language_to_citation(self, n_candidates: int = 30) -> None:
        if self.candidates_language_to_citation is None:
            self.candidates_language_to_citation = LanguageModelScorer.display_top_n(
                self.language_model_data, n=n_candidates
            )

        if self.candidate_scores_language_to_citation is None:
            self.candidate_scores_language_to_citation = LanguageModelScorer.score_top_n(
                self.language_model_data, n=n_candidates
            )

        if self.candidate_ids_language_to_citation is None:
            self.candidate_ids_language_to_citation = self.candidates_language_to_citation.index

    def select_language_to_citation(
        self,
        scoring_feature: ScoringFeature = ScoringFeature.weighted,
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> pd.DataFrame:
        self.set_candidates_language_to_citation(n_candidates)

        return CitationModelScorer.display_top_n(
            self.citation_model_data[self.candidate_ids_language_to_citation],
            scoring_feature,
            n=n_final,
        )

    def score_language_to_citation(
        self,
        scoring_feature: ScoringFeature = ScoringFeature.weighted,
        n_candidates: int = 30,
        n_final: int = 10,
    ) -> float:
        self.set_candidates_language_to_citation(n_candidates)

        return CitationModelScorer.score_top_n(
            self.citation_model_data[self.candidate_ids_language_to_citation],
            scoring_feature,
            n=n_final,
        )


query_document_id = 206594692

documents_authors_labels_citations_most_cited: pd.DataFrame = pd.read_pickle(
    DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
).set_index("document_id")
# NOTE: Remove to evaluate on full data
documents_authors_labels_citations_most_cited = documents_authors_labels_citations_most_cited.head(
    1000
)

bibliographic_coupling_scores_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl
)

co_citation_analysis_scores_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
)


# SECTION: Citation Models
citation_model_data_from_id = CitationModelDataFromId(
    query_document_id=query_document_id,
    documents_data=documents_authors_labels_citations_most_cited.pipe(
        CitationModelScorer.add_feature_rank_cols
    ).pipe(CitationModelScorer.set_missing_publication_dates_to_max_rank),
    co_citation_analysis_scores=co_citation_analysis_scores_most_cited,
    bibliographic_coupling_scores=bibliographic_coupling_scores_most_cited,
)
citation_model_data = citation_model_data_from_id.get_model_data()


# SECTION: TF-IDF
tfidf_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
)
tfidf_data_from_id = LanguageModelDataFromId(
    query_document_id=query_document_id,
    documents_data=documents_authors_labels_citations_most_cited,
    cosine_similarities=tfidf_cosine_similarities_most_cited,
)
tfidf_data = tfidf_data_from_id.get_model_data()

# SECTION: Hybrid Model
tfidf_hybrid_recommender = HybridRecommender(
    citation_model_data=citation_model_data, language_model_data=tfidf_data
)

print(tfidf_hybrid_recommender.score_citation_to_language(n_candidates=30, n_final=30))
print(tfidf_hybrid_recommender.candidate_scores_citation_to_language)
print(tfidf_hybrid_recommender.score_language_to_citation(n_candidates=30, n_final=30))
print(tfidf_hybrid_recommender.candidate_scores_language_to_citation)

tfidf_hybrid_recommender.select_citation_to_language()
tfidf_hybrid_recommender.select_language_to_citation()
