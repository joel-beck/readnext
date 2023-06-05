"""
Compute evaluation metrics and show the best recommendations of either the Citation
model or a single Language Model for a single query document.
"""

import polars as pl

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.metrics import AveragePrecision
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
    documents_data = read_df_from_parquet(DataPaths.merged.documents_data)
    # NOTE: Remove to evaluate on full data
    documents_data = documents_data.head(1000)

    bibliographic_coupling_scores: ScoresFrame = read_df_from_parquet(
        ResultsPaths.citation_models.bibliographic_coupling_scores_parquet
    )

    co_citation_analysis_scores: ScoresFrame = read_df_from_parquet(
        ResultsPaths.citation_models.co_citation_analysis_scores_parquet
    )

    model_data_constructor_plugin = SeenModelDataConstructorPlugin(
        query_d3_document_id, documents_data
    )

    # SECTION: Get Model Data
    # SUBSECTION: Citation Models
    citation_model_data_constructor = CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        co_citation_analysis_scores_frame=co_citation_analysis_scores,
        bibliographic_coupling_scores_frame=bibliographic_coupling_scores,
        constructor_plugin=model_data_constructor_plugin,
    )
    citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)

    print(citation_model_data.query_document)

    # SUBSECTION: TF-IDF
    tfidf_cosine_similarities: pl.DataFrame = read_df_from_parquet(
        ResultsPaths.language_models.tfidf_cosine_similarities_parquet
    )
    tfidf_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarity_scores_frame=tfidf_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    tfidf_data = LanguageModelData.from_constructor(tfidf_data_constructor)
    LanguageModelScorer.display_top_n(tfidf_data, n=20)

    # SUBSECTION: BM25
    bm25_cosine_similarities: pl.DataFrame = read_df_from_parquet(
        ResultsPaths.language_models.bm25_cosine_similarities_parquet
    )
    bm25_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarity_scores_frame=bm25_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    bm25_data = LanguageModelData.from_constructor(bm25_data_constructor)
    LanguageModelScorer.display_top_n(bm25_data, n=20)

    # SUBSECTION: Word2Vec
    word2vec_cosine_similarities: pl.DataFrame = read_df_from_parquet(
        ResultsPaths.language_models.word2vec_cosine_similarities_parquet
    )
    word2vec_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarity_scores_frame=word2vec_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    word2vec_data = LanguageModelData.from_constructor(word2vec_data_constructor)
    LanguageModelScorer.display_top_n(word2vec_data, n=20)

    # SUBSECTION: GloVe
    glove_cosine_similarities: pl.DataFrame = read_df_from_parquet(
        ResultsPaths.language_models.glove_cosine_similarities_parquet
    )
    glove_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarity_scores_frame=glove_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    glove_data = LanguageModelData.from_constructor(glove_data_constructor)
    LanguageModelScorer.display_top_n(glove_data, n=20)

    # SUBSECTION: FastText
    fasttext_cosine_similarities: pl.DataFrame = read_df_from_parquet(
        ResultsPaths.language_models.fasttext_cosine_similarities_parquet
    )
    fasttext_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarity_scores_frame=fasttext_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    fasttext_data = LanguageModelData.from_constructor(fasttext_data_constructor)
    LanguageModelScorer.display_top_n(fasttext_data, n=20)

    # SUBSECTION: BERT
    bert_cosine_similarities: pl.DataFrame = read_df_from_parquet(
        ResultsPaths.language_models.bert_cosine_similarities_parquet
    )
    bert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarity_scores_frame=bert_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    bert_data = LanguageModelData.from_constructor(bert_data_constructor)
    LanguageModelScorer.display_top_n(bert_data, n=20)

    # SUBSECTION: SciBERT
    scibert_cosine_similarities: pl.DataFrame = read_df_from_parquet(
        ResultsPaths.language_models.scibert_cosine_similarities_parquet
    )
    scibert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarity_scores_frame=scibert_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    scibert_data = LanguageModelData.from_constructor(scibert_data_constructor)
    LanguageModelScorer.display_top_n(scibert_data, n=20)

    # SUBSECTION: Longformer
    longformer_cosine_similarities: pl.DataFrame = read_df_from_parquet(
        ResultsPaths.language_models.longformer_cosine_similarities_parquet
    )
    longformer_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_data,
        cosine_similarity_scores_frame=longformer_cosine_similarities,
        constructor_plugin=model_data_constructor_plugin,
    )
    longformer_data = LanguageModelData.from_constructor(longformer_data_constructor)
    LanguageModelScorer.display_top_n(longformer_data, n=20)

    # SECTION: Evaluate Scores
    average_precision_scores = pl.from_records(
        [
            (
                "Publication Date",
                CitationModelScorer.score_top_n(
                    citation_model_data,
                    AveragePrecision(),
                    FeatureWeights(1, 0, 0, 0, 0),
                    n=20,
                ),
            ),
            (
                "Citation Count Document",
                CitationModelScorer.score_top_n(
                    citation_model_data,
                    AveragePrecision(),
                    FeatureWeights(0, 1, 0, 0, 0),
                    n=20,
                ),
            ),
            (
                "Citation Count Author",
                CitationModelScorer.score_top_n(
                    citation_model_data,
                    AveragePrecision(),
                    FeatureWeights(0, 0, 1, 0, 0),
                    n=20,
                ),
            ),
            (
                "Co-Citation Analysis",
                CitationModelScorer.score_top_n(
                    citation_model_data,
                    AveragePrecision(),
                    FeatureWeights(0, 0, 0, 1, 0),
                    n=20,
                ),
            ),
            (
                "Bibliographic Coupling",
                CitationModelScorer.score_top_n(
                    citation_model_data,
                    AveragePrecision(),
                    FeatureWeights(0, 0, 0, 0, 1),
                    n=20,
                ),
            ),
            (
                "Weighted",
                CitationModelScorer.score_top_n(
                    citation_model_data, AveragePrecision(), FeatureWeights(), n=20
                ),
            ),
            (
                "TF-IDF",
                LanguageModelScorer.score_top_n(tfidf_data, AveragePrecision(), n=20),
            ),
            (
                "BM25",
                LanguageModelScorer.score_top_n(bm25_data, AveragePrecision(), n=20),
            ),
            (
                "Word2Vec",
                LanguageModelScorer.score_top_n(word2vec_data, AveragePrecision(), n=20),
            ),
            (
                "GloVe",
                LanguageModelScorer.score_top_n(glove_data, AveragePrecision(), n=20),
            ),
            (
                "FastText",
                LanguageModelScorer.score_top_n(fasttext_data, AveragePrecision(), n=20),
            ),
            (
                "BERT",
                LanguageModelScorer.score_top_n(bert_data, AveragePrecision(), n=20),
            ),
            (
                "SciBERT",
                LanguageModelScorer.score_top_n(scibert_data, AveragePrecision(), n=20),
            ),
        ],
        schema=["Feature", "Average Precision"],
    ).sort(by="Average Precision", descending=True)

    print(average_precision_scores)


if __name__ == "__main__":
    main()
