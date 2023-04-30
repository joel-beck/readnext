import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.modeling import (
    CitationModelDataFromId,
    LanguageModelDataFromId,
)
from readnext.modeling.citation_models import (
    add_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling.hybrid_models import HybridRecommender, HybridScore, compare_hybrid_scores


def compare_hybrid_scores_by_document_id(
    query_document_id: int,
    documents_data: pd.DataFrame,
    co_citation_analysis_scores: pd.DataFrame,
    bibliographic_coupling_scores: pd.DataFrame,
) -> pd.DataFrame:
    # SECTION: Citation Models
    citation_model_data_from_id = CitationModelDataFromId(
        query_document_id=query_document_id,
        documents_data=documents_data.pipe(add_feature_rank_cols).pipe(
            set_missing_publication_dates_to_max_rank
        ),
        co_citation_analysis_scores=co_citation_analysis_scores,
        bibliographic_coupling_scores=bibliographic_coupling_scores,
    )
    citation_model_data = citation_model_data_from_id.get_model_data()

    # SECTION: Language Models
    # SUBSECTION: TF-IDF
    tfidf_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
    )
    tfidf_data_from_id = LanguageModelDataFromId(
        query_document_id=query_document_id,
        documents_data=documents_data,
        cosine_similarities=tfidf_cosine_similarities_most_cited,
    )
    tfidf_data = tfidf_data_from_id.get_model_data()

    # SUBSECTION: Word2Vec
    word2vec_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
    )
    word2vec_data_from_id = LanguageModelDataFromId(
        query_document_id=query_document_id,
        documents_data=documents_data,
        cosine_similarities=word2vec_cosine_similarities_most_cited,
    )
    word2vec_data = word2vec_data_from_id.get_model_data()

    # SUBSECTION: FastText
    fasttext_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )
    fasttext_data_from_id = LanguageModelDataFromId(
        query_document_id=query_document_id,
        documents_data=documents_data,
        cosine_similarities=fasttext_cosine_similarities_most_cited,
    )
    fasttext_data = fasttext_data_from_id.get_model_data()

    # SUBSECTION: BERT
    bert_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl
    )
    bert_data_from_id = LanguageModelDataFromId(
        query_document_id=query_document_id,
        documents_data=documents_data,
        cosine_similarities=bert_cosine_similarities_most_cited,
    )
    bert_data = bert_data_from_id.get_model_data()

    # SUBSECTION: SciBERT
    scibert_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
    )
    scibert_data_from_id = LanguageModelDataFromId(
        query_document_id=query_document_id,
        documents_data=documents_data,
        cosine_similarities=scibert_cosine_similarities_most_cited,
    )
    scibert_data = scibert_data_from_id.get_model_data()

    # SECTION: Hybrid Models
    # SUBSECTION: TF-IDF
    tfidf_hybrid_recommender = HybridRecommender(
        language_model_name="Tf-Idf",
        citation_model_data=citation_model_data,
        language_model_data=tfidf_data,
    )
    tfidf_hybrid_recommender.fit(n_candidates=30, n_final=30)

    tfidf_hybrid_recommender.citation_to_language_recommendations
    tfidf_hybrid_recommender.language_to_citation_recommendations

    tfidf_hybrid_score = HybridScore.from_recommender(tfidf_hybrid_recommender)

    # SUBSECTION: Word2Vec
    word2vec_hybrid_recommender = HybridRecommender(
        language_model_name="Word2Vec",
        citation_model_data=citation_model_data,
        language_model_data=word2vec_data,
    )
    word2vec_hybrid_recommender.fit(n_candidates=30, n_final=30)

    word2vec_hybrid_recommender.citation_to_language_recommendations
    word2vec_hybrid_recommender.language_to_citation_recommendations

    word2vec_hybrid_score = HybridScore.from_recommender(word2vec_hybrid_recommender)

    # SUBSECTION: FastText
    fasttext_hybrid_recommender = HybridRecommender(
        language_model_name="FastText",
        citation_model_data=citation_model_data,
        language_model_data=fasttext_data,
    )
    fasttext_hybrid_recommender.fit(n_candidates=30, n_final=30)

    fasttext_hybrid_recommender.citation_to_language_recommendations
    fasttext_hybrid_recommender.language_to_citation_recommendations

    fasttext_hybrid_score = HybridScore.from_recommender(fasttext_hybrid_recommender)

    # SUBSECTION: BERT
    bert_hybrid_recommender = HybridRecommender(
        language_model_name="BERT",
        citation_model_data=citation_model_data,
        language_model_data=bert_data,
    )
    bert_hybrid_recommender.fit(n_candidates=30, n_final=30)

    bert_hybrid_recommender.citation_to_language_recommendations
    bert_hybrid_recommender.language_to_citation_recommendations

    bert_hybrid_score = HybridScore.from_recommender(bert_hybrid_recommender)

    # SUBSECTION: SciBERT
    scibert_hybrid_recommender = HybridRecommender(
        language_model_name="SciBERT",
        citation_model_data=citation_model_data,
        language_model_data=scibert_data,
    )
    scibert_hybrid_recommender.fit(n_candidates=30, n_final=30)

    scibert_hybrid_recommender.citation_to_language_recommendations
    scibert_hybrid_recommender.language_to_citation_recommendations

    scibert_hybrid_score = HybridScore.from_recommender(scibert_hybrid_recommender)

    # SECTION: Compare Scores
    return (
        compare_hybrid_scores(
            tfidf_hybrid_score,
            word2vec_hybrid_score,
            fasttext_hybrid_score,
            bert_hybrid_score,
            scibert_hybrid_score,
        )
        .assign(query_document_id=query_document_id)
        .set_index("query_document_id")
        .rename_axis(index="Query Document ID")
    )


def main() -> None:
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

    # add query document ids to index
    average_precision_scores = pd.concat(
        [
            compare_hybrid_scores_by_document_id(
                query_document_id,
                documents_authors_labels_citations_most_cited,
                co_citation_analysis_scores_most_cited,
                bibliographic_coupling_scores_most_cited,
            )
            for query_document_id in documents_authors_labels_citations_most_cited.head(
                5
            ).index.to_numpy()
        ]
    )

    # add best score column for each query document id
    average_precision_scores["Best Score"] = average_precision_scores.select_dtypes(
        include="number"
    ).max(axis=1)

    # filter best language model for each query document id
    average_precision_scores.sort_values("Best Score", ascending=False).reset_index(
        drop=False
    ).drop_duplicates(subset="Query Document ID").reset_index(drop=True)


if __name__ == "__main__":
    main()
