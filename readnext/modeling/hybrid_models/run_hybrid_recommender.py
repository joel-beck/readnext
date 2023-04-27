import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation import CitationModelScorer
from readnext.modeling import (
    CitationModelDataFromId,
    LanguageModelDataFromId,
)
from readnext.modeling.hybrid_models import HybridRecommender, HybridScores


def main() -> None:
    query_document_id = 206594692

    documents_authors_labels_citations_most_cited: pd.DataFrame = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    ).set_index("document_id")
    # NOTE: Remove to evaluate on full data
    documents_authors_labels_citations_most_cited = (
        documents_authors_labels_citations_most_cited.head(1000)
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

    # SECTION: Language Models
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

    # SUBSECTION: FastText
    fasttext_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )
    fasttext_data_from_id = LanguageModelDataFromId(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=fasttext_cosine_similarities_most_cited,
    )
    fasttext_data = fasttext_data_from_id.get_model_data()

    # SECTION: Hybrid Models
    # SUBSECTION: TF-IDF
    tfidf_hybrid_recommender = HybridRecommender(
        citation_model_data=citation_model_data, language_model_data=tfidf_data
    )

    tfidf_hybrid_scores = HybridScores(
        citation_to_language=tfidf_hybrid_recommender.score_citation_to_language(
            n_candidates=30, n_final=30
        ),
        citation_to_language_candidates=tfidf_hybrid_recommender.citation_to_language_candidate_scores,
        language_to_citation=tfidf_hybrid_recommender.score_language_to_citation(
            n_candidates=30, n_final=30
        ),
        language_to_citation_candidates=tfidf_hybrid_recommender.language_to_citation_candidate_scores,
    )

    print(tfidf_hybrid_scores)

    tfidf_hybrid_recommender.select_citation_to_language()
    tfidf_hybrid_recommender.select_language_to_citation()

    # SUBSECTION: FastText
    fasttext_hybrid_recommender = HybridRecommender(
        citation_model_data=citation_model_data, language_model_data=fasttext_data
    )

    fasttext_hybrid_scores = HybridScores(
        citation_to_language=fasttext_hybrid_recommender.score_citation_to_language(
            n_candidates=30, n_final=30
        ),
        citation_to_language_candidates=fasttext_hybrid_recommender.citation_to_language_candidate_scores,
        language_to_citation=fasttext_hybrid_recommender.score_language_to_citation(
            n_candidates=30, n_final=30
        ),
        language_to_citation_candidates=fasttext_hybrid_recommender.language_to_citation_candidate_scores,
    )

    print(fasttext_hybrid_scores)


if __name__ == "__main__":
    main()
