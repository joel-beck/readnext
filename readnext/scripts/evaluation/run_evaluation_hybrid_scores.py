"""
Compute scores and select the best recommendations from the hybrid recommender model in
both hybrid recommender component orders for a given query document.
"""

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.data import (
    add_citation_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.evaluation.metrics import AveragePrecision
from readnext.evaluation.scoring import HybridScore, HybridScorer, compare_hybrid_scores
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.utils import load_df_from_pickle


def compare_hybrid_scores_by_document_id(
    query_d3_document_id: int,
    documents_data: pd.DataFrame,
    co_citation_analysis_scores: pd.DataFrame,
    bibliographic_coupling_scores: pd.DataFrame,
) -> pd.DataFrame:
    # SECTION: Citation Models
    citation_model_data_constructor = CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data.pipe(add_citation_feature_rank_cols).pipe(
            set_missing_publication_dates_to_max_rank
        ),
        co_citation_analysis_scores=co_citation_analysis_scores,
        bibliographic_coupling_scores=bibliographic_coupling_scores,
    )
    citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)

    # SECTION: Language Models
    # SUBSECTION: TF-IDF

    tfidf_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
    )
    tfidf_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarities=tfidf_cosine_similarities_most_cited,
    )
    tfidf_data = LanguageModelData.from_constructor(tfidf_data_constructor)

    # SUBSECTION: BM25
    bm25_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.bm25_cosine_similarities_most_cited_pkl
    )
    bm25_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarities=bm25_cosine_similarities_most_cited,
    )
    bm25_data = LanguageModelData.from_constructor(bm25_data_constructor)

    # SUBSECTION: Word2Vec
    word2vec_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
    )
    word2vec_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarities=word2vec_cosine_similarities_most_cited,
    )
    word2vec_data = LanguageModelData.from_constructor(word2vec_data_constructor)

    # SUBSECTION: GloVe
    glove_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.glove_cosine_similarities_most_cited_pkl
    )
    glove_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarities=glove_cosine_similarities_most_cited,
    )
    glove_data = LanguageModelData.from_constructor(glove_data_constructor)

    # SUBSECTION: FastText
    fasttext_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )
    fasttext_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarities=fasttext_cosine_similarities_most_cited,
    )
    fasttext_data = LanguageModelData.from_constructor(fasttext_data_constructor)

    # SUBSECTION: BERT
    bert_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl
    )
    bert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarities=bert_cosine_similarities_most_cited,
    )
    bert_data = LanguageModelData.from_constructor(bert_data_constructor)

    # SUBSECTION: SciBERT
    scibert_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
    )
    scibert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarities=scibert_cosine_similarities_most_cited,
    )
    scibert_data = LanguageModelData.from_constructor(scibert_data_constructor)

    # SUBSECTION: Longformer
    longformer_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.longformer_cosine_similarities_most_cited_pkl
    )
    longformer_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarities=longformer_cosine_similarities_most_cited,
    )
    longformer_data = LanguageModelData.from_constructor(longformer_data_constructor)

    # SECTION: Hybrid Models
    # SUBSECTION: TF-IDF
    tfidf_hybrid_scorer = HybridScorer(
        language_model_name="Tf-Idf",
        citation_model_data=citation_model_data,
        language_model_data=tfidf_data,
    )
    tfidf_hybrid_scorer.fit(AveragePrecision(), n_candidates=30, n_final=30)

    tfidf_hybrid_scorer.citation_to_language_recommendations
    tfidf_hybrid_scorer.language_to_citation_recommendations

    tfidf_hybrid_score = HybridScore.from_scorer(tfidf_hybrid_scorer)

    # SUBSECTION: BM25
    bm25_hybrid_scorer = HybridScorer(
        language_model_name="Tf-Idf",
        citation_model_data=citation_model_data,
        language_model_data=bm25_data,
    )
    bm25_hybrid_scorer.fit(AveragePrecision(), n_candidates=30, n_final=30)

    bm25_hybrid_scorer.citation_to_language_recommendations
    bm25_hybrid_scorer.language_to_citation_recommendations

    bm25_hybrid_score = HybridScore.from_scorer(bm25_hybrid_scorer)

    # SUBSECTION: Word2Vec
    word2vec_hybrid_scorer = HybridScorer(
        language_model_name="Word2Vec",
        citation_model_data=citation_model_data,
        language_model_data=word2vec_data,
    )
    word2vec_hybrid_scorer.fit(AveragePrecision(), n_candidates=30, n_final=30)

    word2vec_hybrid_scorer.citation_to_language_recommendations
    word2vec_hybrid_scorer.language_to_citation_recommendations

    word2vec_hybrid_score = HybridScore.from_scorer(word2vec_hybrid_scorer)

    # SUBSECTION: GloVe
    glove_hybrid_scorer = HybridScorer(
        language_model_name="GloVe",
        citation_model_data=citation_model_data,
        language_model_data=glove_data,
    )
    glove_hybrid_scorer.fit(AveragePrecision(), n_candidates=30, n_final=30)

    glove_hybrid_scorer.citation_to_language_recommendations
    glove_hybrid_scorer.language_to_citation_recommendations

    glove_hybrid_score = HybridScore.from_scorer(glove_hybrid_scorer)

    # SUBSECTION: FastText
    fasttext_hybrid_scorer = HybridScorer(
        language_model_name="FastText",
        citation_model_data=citation_model_data,
        language_model_data=fasttext_data,
    )
    fasttext_hybrid_scorer.fit(AveragePrecision(), n_candidates=30, n_final=30)

    fasttext_hybrid_scorer.citation_to_language_recommendations
    fasttext_hybrid_scorer.language_to_citation_recommendations

    fasttext_hybrid_score = HybridScore.from_scorer(fasttext_hybrid_scorer)

    # SUBSECTION: BERT
    bert_hybrid_scorer = HybridScorer(
        language_model_name="BERT",
        citation_model_data=citation_model_data,
        language_model_data=bert_data,
    )
    bert_hybrid_scorer.fit(AveragePrecision(), n_candidates=30, n_final=30)

    bert_hybrid_scorer.citation_to_language_recommendations
    bert_hybrid_scorer.language_to_citation_recommendations

    bert_hybrid_score = HybridScore.from_scorer(bert_hybrid_scorer)

    # SUBSECTION: SciBERT
    scibert_hybrid_scorer = HybridScorer(
        language_model_name="SciBERT",
        citation_model_data=citation_model_data,
        language_model_data=scibert_data,
    )
    scibert_hybrid_scorer.fit(AveragePrecision(), n_candidates=30, n_final=30)

    scibert_hybrid_scorer.citation_to_language_recommendations
    scibert_hybrid_scorer.language_to_citation_recommendations

    scibert_hybrid_score = HybridScore.from_scorer(scibert_hybrid_scorer)

    # SUBSECTION: Longformer
    longformer_hybrid_scorer = HybridScorer(
        language_model_name="Longformer",
        citation_model_data=citation_model_data,
        language_model_data=longformer_data,
    )
    longformer_hybrid_scorer.fit(AveragePrecision(), n_candidates=30, n_final=30)

    longformer_hybrid_scorer.citation_to_language_recommendations
    longformer_hybrid_scorer.language_to_citation_recommendations

    longformer_hybrid_score = HybridScore.from_scorer(longformer_hybrid_scorer)

    # SECTION: Compare Scores
    return (
        compare_hybrid_scores(
            tfidf_hybrid_score,
            bm25_hybrid_score,
            word2vec_hybrid_score,
            glove_hybrid_score,
            fasttext_hybrid_score,
            bert_hybrid_score,
            scibert_hybrid_score,
            longformer_hybrid_score,
        )
        .assign(query_d3_document_id=query_d3_document_id)
        .set_index("query_d3_document_id")
        .rename_axis(index="Query Document ID")
    )


def main() -> None:
    documents_authors_labels_citations_most_cited: pd.DataFrame = load_df_from_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )
    # NOTE: Remove to evaluate on full data
    documents_authors_labels_citations_most_cited = (
        documents_authors_labels_citations_most_cited.head(1000)
    )

    bibliographic_coupling_scores_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl
    )

    co_citation_analysis_scores_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
    )

    # add query document ids to index
    average_precision_scores = pd.concat(
        [
            compare_hybrid_scores_by_document_id(
                query_d3_document_id,
                documents_authors_labels_citations_most_cited,
                co_citation_analysis_scores_most_cited,
                bibliographic_coupling_scores_most_cited,
            )
            for query_d3_document_id in documents_authors_labels_citations_most_cited.head(
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
