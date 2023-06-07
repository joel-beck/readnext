"""
Compute scores and select the best recommendations from the hybrid recommender model in
both hybrid recommender component orders for a given query document.
"""

import polars as pl

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.metrics import AveragePrecision
from readnext.evaluation.scoring import HybridScore, HybridScorer, compare_hybrid_scores
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
    SeenModelDataConstructorPlugin,
)
from readnext.utils import DocumentsFrame, ScoresFrame, read_df_from_parquet


def compare_hybrid_scores_by_document_id(
    query_d3_document_id: int,
    documents_frame: DocumentsFrame,
    co_citation_analysis_scores: ScoresFrame,
    bibliographic_coupling_scores: ScoresFrame,
) -> pl.DataFrame:
    # SECTION: Citation Models
    model_data_constructor_plugin = SeenModelDataConstructorPlugin(
        query_d3_document_id, documents_frame
    )

    citation_model_data_constructor = CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        co_citation_analysis_scores_frame=co_citation_analysis_scores,
        bibliographic_coupling_scores_frame=bibliographic_coupling_scores,
        constructor_plugin=model_data_constructor_plugin,
    )
    citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)

    # SECTION: Language Models
    # SUBSECTION: TF-IDF
    tfidf_cosine_similarities: ScoresFrame = read_df_from_parquet(
        ResultsPaths.language_models.tfidf_cosine_similarities_parquet
    )
    tfidf_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        cosine_similarity_scores_frame=tfidf_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    tfidf_data = LanguageModelData.from_constructor(tfidf_data_constructor)

    # SUBSECTION: BM25
    bm25_cosine_similarities: ScoresFrame = read_df_from_parquet(
        ResultsPaths.language_models.bm25_cosine_similarities_parquet
    )
    bm25_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        cosine_similarity_scores_frame=bm25_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    bm25_data = LanguageModelData.from_constructor(bm25_data_constructor)

    # SUBSECTION: Word2Vec
    word2vec_cosine_similarities: ScoresFrame = read_df_from_parquet(
        ResultsPaths.language_models.word2vec_cosine_similarities_parquet
    )
    word2vec_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        cosine_similarity_scores_frame=word2vec_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    word2vec_data = LanguageModelData.from_constructor(word2vec_data_constructor)

    # SUBSECTION: GloVe
    glove_cosine_similarities: ScoresFrame = read_df_from_parquet(
        ResultsPaths.language_models.glove_cosine_similarities_parquet
    )
    glove_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        cosine_similarity_scores_frame=glove_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    glove_data = LanguageModelData.from_constructor(glove_data_constructor)

    # SUBSECTION: FastText
    fasttext_cosine_similarities: ScoresFrame = read_df_from_parquet(
        ResultsPaths.language_models.fasttext_cosine_similarities_parquet
    )
    fasttext_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        cosine_similarity_scores_frame=fasttext_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    fasttext_data = LanguageModelData.from_constructor(fasttext_data_constructor)

    # SUBSECTION: BERT
    bert_cosine_similarities: ScoresFrame = read_df_from_parquet(
        ResultsPaths.language_models.bert_cosine_similarities_parquet
    )
    bert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        cosine_similarity_scores_frame=bert_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    bert_data = LanguageModelData.from_constructor(bert_data_constructor)

    # SUBSECTION: SciBERT
    scibert_cosine_similarities: ScoresFrame = read_df_from_parquet(
        ResultsPaths.language_models.scibert_cosine_similarities_parquet
    )
    scibert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        cosine_similarity_scores_frame=scibert_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    scibert_data = LanguageModelData.from_constructor(scibert_data_constructor)

    # SUBSECTION: Longformer
    longformer_cosine_similarities: ScoresFrame = read_df_from_parquet(
        ResultsPaths.language_models.longformer_cosine_similarities_parquet
    )
    longformer_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        cosine_similarity_scores_frame=longformer_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    longformer_data = LanguageModelData.from_constructor(longformer_data_constructor)

    # SECTION: Hybrid Models
    # SUBSECTION: TF-IDF
    tfidf_hybrid_scorer = HybridScorer(
        language_model_name="Tf-Idf",
        citation_model_data=citation_model_data,
        language_model_data=tfidf_data,
    )
    tfidf_hybrid_scorer.fit(AveragePrecision(), n_candidates=20, n_final=20)

    tfidf_hybrid_scorer.citation_to_language_recommendations
    tfidf_hybrid_scorer.language_to_citation_recommendations

    tfidf_hybrid_score = HybridScore.from_scorer(tfidf_hybrid_scorer)

    # SUBSECTION: BM25
    bm25_hybrid_scorer = HybridScorer(
        language_model_name="Tf-Idf",
        citation_model_data=citation_model_data,
        language_model_data=bm25_data,
    )
    bm25_hybrid_scorer.fit(AveragePrecision(), n_candidates=20, n_final=20)

    bm25_hybrid_scorer.citation_to_language_recommendations
    bm25_hybrid_scorer.language_to_citation_recommendations

    bm25_hybrid_score = HybridScore.from_scorer(bm25_hybrid_scorer)

    # SUBSECTION: Word2Vec
    word2vec_hybrid_scorer = HybridScorer(
        language_model_name="Word2Vec",
        citation_model_data=citation_model_data,
        language_model_data=word2vec_data,
    )
    word2vec_hybrid_scorer.fit(AveragePrecision(), n_candidates=20, n_final=20)

    word2vec_hybrid_scorer.citation_to_language_recommendations
    word2vec_hybrid_scorer.language_to_citation_recommendations

    word2vec_hybrid_score = HybridScore.from_scorer(word2vec_hybrid_scorer)

    # SUBSECTION: GloVe
    glove_hybrid_scorer = HybridScorer(
        language_model_name="GloVe",
        citation_model_data=citation_model_data,
        language_model_data=glove_data,
    )
    glove_hybrid_scorer.fit(AveragePrecision(), n_candidates=20, n_final=20)

    glove_hybrid_scorer.citation_to_language_recommendations
    glove_hybrid_scorer.language_to_citation_recommendations

    glove_hybrid_score = HybridScore.from_scorer(glove_hybrid_scorer)

    # SUBSECTION: FastText
    fasttext_hybrid_scorer = HybridScorer(
        language_model_name="FastText",
        citation_model_data=citation_model_data,
        language_model_data=fasttext_data,
    )
    fasttext_hybrid_scorer.fit(AveragePrecision(), n_candidates=20, n_final=20)

    fasttext_hybrid_scorer.citation_to_language_recommendations
    fasttext_hybrid_scorer.language_to_citation_recommendations

    fasttext_hybrid_score = HybridScore.from_scorer(fasttext_hybrid_scorer)

    # SUBSECTION: BERT
    bert_hybrid_scorer = HybridScorer(
        language_model_name="BERT",
        citation_model_data=citation_model_data,
        language_model_data=bert_data,
    )
    bert_hybrid_scorer.fit(AveragePrecision(), n_candidates=20, n_final=20)

    bert_hybrid_scorer.citation_to_language_recommendations
    bert_hybrid_scorer.language_to_citation_recommendations

    bert_hybrid_score = HybridScore.from_scorer(bert_hybrid_scorer)

    # SUBSECTION: SciBERT
    scibert_hybrid_scorer = HybridScorer(
        language_model_name="SciBERT",
        citation_model_data=citation_model_data,
        language_model_data=scibert_data,
    )
    scibert_hybrid_scorer.fit(AveragePrecision(), n_candidates=20, n_final=20)

    scibert_hybrid_scorer.citation_to_language_recommendations
    scibert_hybrid_scorer.language_to_citation_recommendations

    scibert_hybrid_score = HybridScore.from_scorer(scibert_hybrid_scorer)

    # SUBSECTION: Longformer
    longformer_hybrid_scorer = HybridScorer(
        language_model_name="Longformer",
        citation_model_data=citation_model_data,
        language_model_data=longformer_data,
    )
    longformer_hybrid_scorer.fit(AveragePrecision(), n_candidates=20, n_final=20)

    longformer_hybrid_scorer.citation_to_language_recommendations
    longformer_hybrid_scorer.language_to_citation_recommendations

    longformer_hybrid_score = HybridScore.from_scorer(longformer_hybrid_scorer)

    # SECTION: Compare Scores
    return compare_hybrid_scores(
        tfidf_hybrid_score,
        bm25_hybrid_score,
        word2vec_hybrid_score,
        glove_hybrid_score,
        fasttext_hybrid_score,
        bert_hybrid_score,
        scibert_hybrid_score,
        longformer_hybrid_score,
    ).insert_at_idx(0, pl.Series("Query Document ID", [query_d3_document_id] * 8))


def main() -> None:
    documents_frame: DocumentsFrame = read_df_from_parquet(DataPaths.merged.documents_frame)
    # query_d3_document_id = documents_frame["d3_document_id"][0]
    documents_frame = documents_frame.head(1000)

    bibliographic_coupling_scores: ScoresFrame = read_df_from_parquet(
        ResultsPaths.citation_models.bibliographic_coupling_scores_parquet
    )

    co_citation_analysis_scores: ScoresFrame = read_df_from_parquet(
        ResultsPaths.citation_models.co_citation_analysis_scores_parquet
    )

    # add query document ids to index
    average_precision_scores = pl.concat(
        [
            compare_hybrid_scores_by_document_id(
                query_d3_document_id,
                documents_frame,
                co_citation_analysis_scores,
                bibliographic_coupling_scores,
            )
            for query_d3_document_id in documents_frame["d3_document_id"].head(5)
        ]
    ).with_columns(pl.max(pl.exclude(["Query Document ID", "Language Model"])).alias("Best Score"))

    print(average_precision_scores)

    # filter best language model for each query document id
    average_precision_scores.sort(by="Best Score", descending=True).unique(
        subset="Query Document ID"
    )


if __name__ == "__main__":
    main()
