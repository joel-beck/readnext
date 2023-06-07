"""
Compute evaluation metrics and show the best recommendations of either the Citation
model or a single Language Model for a single query document.
"""

import polars as pl

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.metrics import CountUniqueLabels
from readnext.evaluation.scoring import CitationModelScorer, FeatureWeights, LanguageModelScorer
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
    SeenModelDataConstructorPlugin,
)
from readnext.utils import ScoresFrame, read_df_from_parquet


def main() -> None:
    # evaluation for a single query document
    query_d3_document_id = 13756489

    # SECTION: Get Raw Data
    documents_frame = read_df_from_parquet(DataPaths.merged.documents_frame)

    bibliographic_coupling_scores: ScoresFrame = read_df_from_parquet(
        ResultsPaths.citation_models.bibliographic_coupling_scores_parquet
    )

    co_citation_analysis_scores: ScoresFrame = read_df_from_parquet(
        ResultsPaths.citation_models.co_citation_analysis_scores_parquet
    )

    model_data_constructor_plugin = SeenModelDataConstructorPlugin(
        query_d3_document_id, documents_frame
    )

    # SECTION: Get Model Data
    # SUBSECTION: Citation Models
    citation_model_data_constructor = CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_frame=documents_frame,
        co_citation_analysis_scores_frame=co_citation_analysis_scores,
        bibliographic_coupling_scores_frame=bibliographic_coupling_scores,
        constructor_plugin=model_data_constructor_plugin,
    )
    citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)
    citation_model_scorer = CitationModelScorer(citation_model_data)

    print(citation_model_data.query_document)

    citation_model_scorer.display_top_n()

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
    tfidf_language_model_scorer = LanguageModelScorer(tfidf_data)

    tfidf_language_model_scorer.display_top_n()

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
    bm25_language_model_scorer = LanguageModelScorer(bm25_data)

    bm25_language_model_scorer.display_top_n()

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
    word2vec_language_model_scorer = LanguageModelScorer(word2vec_data)

    word2vec_language_model_scorer.display_top_n()

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
    glove_language_model_scorer = LanguageModelScorer(glove_data)

    glove_language_model_scorer.display_top_n()

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
    fasttext_language_model_scorer = LanguageModelScorer(fasttext_data)

    fasttext_language_model_scorer.display_top_n()

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
    bert_language_model_scorer = LanguageModelScorer(bert_data)

    bert_language_model_scorer.display_top_n()

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
    scibert_language_model_scorer = LanguageModelScorer(scibert_data)

    scibert_language_model_scorer.display_top_n()

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
    longformer_language_model_scorer = LanguageModelScorer(longformer_data)

    longformer_language_model_scorer.display_top_n()

    # SECTION: Evaluate Scores
    average_precision_scores = pl.from_records(
        [
            (
                "Publication Date",
                citation_model_scorer.score_top_n(
                    CountUniqueLabels(),
                    FeatureWeights(1, 0, 0, 0, 0),
                ),
            ),
            (
                "Citation Count Document",
                citation_model_scorer.score_top_n(
                    CountUniqueLabels(),
                    FeatureWeights(0, 1, 0, 0, 0),
                ),
            ),
            (
                "Citation Count Author",
                citation_model_scorer.score_top_n(
                    CountUniqueLabels(),
                    FeatureWeights(0, 0, 1, 0, 0),
                ),
            ),
            (
                "Co-Citation Analysis",
                citation_model_scorer.score_top_n(
                    CountUniqueLabels(),
                    FeatureWeights(0, 0, 0, 1, 0),
                ),
            ),
            (
                "Bibliographic Coupling",
                citation_model_scorer.score_top_n(
                    CountUniqueLabels(),
                    FeatureWeights(0, 0, 0, 0, 1),
                ),
            ),
            (
                "Weighted",
                citation_model_scorer.score_top_n(CountUniqueLabels()),
            ),
            (
                "TF-IDF",
                tfidf_language_model_scorer.score_top_n(CountUniqueLabels()),
            ),
            (
                "BM25",
                bm25_language_model_scorer.score_top_n(CountUniqueLabels()),
            ),
            (
                "Word2Vec",
                word2vec_language_model_scorer.score_top_n(CountUniqueLabels()),
            ),
            (
                "GloVe",
                glove_language_model_scorer.score_top_n(CountUniqueLabels()),
            ),
            (
                "FastText",
                fasttext_language_model_scorer.score_top_n(CountUniqueLabels()),
            ),
            (
                "BERT",
                bert_language_model_scorer.score_top_n(CountUniqueLabels()),
            ),
            (
                "SciBERT",
                scibert_language_model_scorer.score_top_n(CountUniqueLabels()),
            ),
            (
                "Longformer",
                longformer_language_model_scorer.score_top_n(CountUniqueLabels()),
            ),
        ],
        schema=["Feature", "Average Precision"],
    ).sort(by="Average Precision", descending=True)

    print(average_precision_scores)


if __name__ == "__main__":
    main()
