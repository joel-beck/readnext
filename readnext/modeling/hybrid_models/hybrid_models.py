from dataclasses import dataclass

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation import CitationModelScorer, LanguageModelScorer, ScoringFeature
from readnext.modeling import CitationModelDataFromId, LanguageModelDataFromId, ModelDataFromId

query_document_id = 206594692

# SECTION: Get Raw Data
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

citation_model_scorer = CitationModelScorer()
language_model_scorer = LanguageModelScorer()

# SECTION: Get Model Data
# SUBSECTION: Citation Models
citation_model_data_from_id = CitationModelDataFromId(
    query_document_id=query_document_id,
    documents_data=documents_authors_labels_citations_most_cited.pipe(
        citation_model_scorer.add_feature_rank_cols
    ).pipe(citation_model_scorer.set_missing_publication_dates_to_max_rank),
    co_citation_analysis_scores=co_citation_analysis_scores_most_cited,
    bibliographic_coupling_scores=bibliographic_coupling_scores_most_cited,
)
first_model_data = citation_model_data_from_id.get_model_data()

print(first_model_data.query_document)

# SUBSECTION: TF-IDF
tfidf_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
)
tfidf_data_from_id = LanguageModelDataFromId(
    query_document_id=query_document_id,
    documents_data=documents_authors_labels_citations_most_cited,
    cosine_similarities=tfidf_cosine_similarities_most_cited,
)
tfidf_data = tfidf_data_from_id.get_model_data()
language_model_scorer.display_top_n(tfidf_data, n=10)

# SECTION: Experimenting for Hybrid Model
citation_model_data_from_id = citation_model_data_from_id
citation_model_data = citation_model_data_from_id.get_model_data()

language_model_data_from_id = tfidf_data_from_id
language_model_data = language_model_data_from_id.get_model_data()

# SUBSECTION: Citation to Language
# Candidate Documents
candidate_ids = language_model_scorer.display_top_n(language_model_data, n=30).index
# Score of Candidate Documents
language_model_scorer.score_top_n(language_model_data, n=30)

# Recommendations of Hybrid Recommender
citation_model_scorer.display_top_n(
    citation_model_data[candidate_ids], scoring_feature=ScoringFeature.weighted, n=10
)
# Final Score of Hybrid Recommender
citation_model_scorer.score_top_n(
    citation_model_data[candidate_ids], scoring_feature=ScoringFeature.weighted, n=30
)

# SUBSECTION: Language to Citation
# Candidate Documents
candidate_ids = citation_model_scorer.display_top_n(
    citation_model_data, scoring_feature=ScoringFeature.weighted, n=30
).index
# Score of Candidate Documents
citation_model_scorer.score_top_n(
    citation_model_data, scoring_feature=ScoringFeature.weighted, n=30
)

# Recommendations of Hybrid Recommender
language_model_scorer.display_top_n(language_model_data[candidate_ids], n=10)
# Final Score of Hybrid Recommender
language_model_scorer.score_top_n(language_model_data[candidate_ids], n=30)


@dataclass
class HybridRecommender:
    query_document_id: int
    language_model_from_id: LanguageModelDataFromId
    citation_model_data_from_id: ModelDataFromId
    scoring_feature: ScoringFeature
    n_candidates: int = 30
    n_final: int = 10

    def citation_to_language(self) -> None:
        pass

    def language_to_citation(self) -> None:
        pass
