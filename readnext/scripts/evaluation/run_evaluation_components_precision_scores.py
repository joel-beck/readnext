"""
Compute evaluation metrics and show the best recommendations of either the Citation
model or a single Language Model for a single query document.
"""

import polars as pl

from readnext.config import DataPaths, ResultsPaths
from readnext.data import (
    add_citation_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.evaluation.metrics import AveragePrecision
from readnext.evaluation.scoring import CitationModelScorer, FeatureWeights, LanguageModelScorer
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.utils import read_df_from_parquet, read_scores_frame_from_parquet


def main() -> None:
    # evaluation for a single query document
    query_d3_document_id = 13756489

    # SECTION: Get Raw Data
    documents_authors_labels_citations_most_cited: pl.DataFrame = read_df_from_parquet(
        DataPaths.merged.documents_authors_labels_citations_most_cited_parquet
    )
    # NOTE: Remove to evaluate on full data
    documents_authors_labels_citations_most_cited = (
        documents_authors_labels_citations_most_cited.head(2000)
    )

    bibliographic_coupling_scores_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_parquet
    )

    co_citation_analysis_scores_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_parquet
    )

    # SECTION: Get Model Data
    # SUBSECTION: Citation Models
    citation_model_data_constructor = CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited.pipe(
            add_citation_feature_rank_cols
        ).pipe(set_missing_publication_dates_to_max_rank),
        co_citation_analysis_scores=co_citation_analysis_scores_most_cited,
        bibliographic_coupling_scores=bibliographic_coupling_scores_most_cited,
    )
    citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)

    print(citation_model_data.query_document)

    # SUBSECTION: TF-IDF
    tfidf_cosine_similarities_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_parquet
    )
    tfidf_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=tfidf_cosine_similarities_most_cited,
    )
    tfidf_data = LanguageModelData.from_constructor(tfidf_data_constructor)
    LanguageModelScorer.display_top_n(tfidf_data, n=20)

    # SUBSECTION: BM25
    bm25_cosine_similarities_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.language_models.bm25_cosine_similarities_most_cited_parquet
    )
    bm25_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=bm25_cosine_similarities_most_cited,
    )
    bm25_data = LanguageModelData.from_constructor(bm25_data_constructor)
    LanguageModelScorer.display_top_n(bm25_data, n=20)

    # SUBSECTION: Word2Vec
    word2vec_cosine_similarities_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_parquet
    )
    word2vec_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=word2vec_cosine_similarities_most_cited,
    )
    word2vec_data = LanguageModelData.from_constructor(word2vec_data_constructor)
    LanguageModelScorer.display_top_n(word2vec_data, n=20)

    # SUBSECTION: GloVe
    glove_cosine_similarities_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.language_models.glove_cosine_similarities_most_cited_parquet
    )
    glove_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=glove_cosine_similarities_most_cited,
    )
    glove_data = LanguageModelData.from_constructor(glove_data_constructor)
    LanguageModelScorer.display_top_n(glove_data, n=20)

    # SUBSECTION: FastText
    fasttext_cosine_similarities_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_parquet
    )
    fasttext_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=fasttext_cosine_similarities_most_cited,
    )
    fasttext_data = LanguageModelData.from_constructor(fasttext_data_constructor)
    LanguageModelScorer.display_top_n(fasttext_data, n=20)

    # SUBSECTION: BERT
    bert_cosine_similarities_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_parquet
    )
    bert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=bert_cosine_similarities_most_cited,
    )
    bert_data = LanguageModelData.from_constructor(bert_data_constructor)
    LanguageModelScorer.display_top_n(bert_data, n=20)

    # SUBSECTION: SciBERT
    scibert_cosine_similarities_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_parquet
    )
    scibert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=scibert_cosine_similarities_most_cited,
    )
    scibert_data = LanguageModelData.from_constructor(scibert_data_constructor)
    LanguageModelScorer.display_top_n(scibert_data, n=20)

    # SUBSECTION: Longformer
    longformer_cosine_similarities_most_cited: pl.DataFrame = read_scores_frame_from_parquet(
        ResultsPaths.language_models.longformer_cosine_similarities_most_cited_parquet
    )
    longformer_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=longformer_cosine_similarities_most_cited,
    )
    longformer_data = LanguageModelData.from_constructor(longformer_data_constructor)
    LanguageModelScorer.display_top_n(longformer_data, n=20)

    # SECTION: Evaluate Scores
    average_precision_scores = pl.DataFrame(
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
        orient="col",
    ).sort(by="Average Precision", descending=True)

    print(average_precision_scores)


if __name__ == "__main__":
    main()
