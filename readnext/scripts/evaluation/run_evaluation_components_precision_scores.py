"""
Compute evaluation metrics and show the best recommendations of either the Citation
model or a single Language Model for a single query document.
"""

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.metrics import AveragePrecision
from readnext.evaluation.scoring import CitationModelScorer, FeatureWeights, LanguageModelScorer
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.modeling.citation_models import (
    add_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.utils import load_df_from_pickle


def main() -> None:
    # evaluation for a single query document
    query_d3_document_id = 206594692

    # SECTION: Get Raw Data
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

    # SECTION: Get Model Data
    # SUBSECTION: Citation Models
    citation_model_data_constructor = CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited.pipe(
            add_feature_rank_cols
        ).pipe(set_missing_publication_dates_to_max_rank),
        co_citation_analysis_scores=co_citation_analysis_scores_most_cited,
        bibliographic_coupling_scores=bibliographic_coupling_scores_most_cited,
    )
    citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)

    print(citation_model_data.query_document)

    # SUBSECTION: TF-IDF
    tfidf_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
    )
    tfidf_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=tfidf_cosine_similarities_most_cited,
    )
    tfidf_data = LanguageModelData.from_constructor(tfidf_data_constructor)
    LanguageModelScorer.display_top_n(tfidf_data, n=10)

    # SUBSECTION: BM25
    bm25_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.bm25_cosine_similarities_most_cited_pkl
    )
    bm25_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=bm25_cosine_similarities_most_cited,
    )
    bm25_data = LanguageModelData.from_constructor(bm25_data_constructor)
    LanguageModelScorer.display_top_n(bm25_data, n=10)

    # SUBSECTION: Word2Vec
    word2vec_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
    )
    word2vec_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=word2vec_cosine_similarities_most_cited,
    )
    word2vec_data = LanguageModelData.from_constructor(word2vec_data_constructor)
    LanguageModelScorer.display_top_n(word2vec_data, n=10)

    # SUBSECTION: GloVe
    glove_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.glove_cosine_similarities_most_cited_pkl
    )
    glove_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=glove_cosine_similarities_most_cited,
    )
    glove_data = LanguageModelData.from_constructor(glove_data_constructor)
    LanguageModelScorer.display_top_n(glove_data, n=10)

    # SUBSECTION: FastText
    fasttext_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )
    fasttext_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=fasttext_cosine_similarities_most_cited,
    )
    fasttext_data = LanguageModelData.from_constructor(fasttext_data_constructor)
    LanguageModelScorer.display_top_n(fasttext_data, n=10)

    # SUBSECTION: BERT
    bert_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl
    )
    bert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=bert_cosine_similarities_most_cited,
    )
    bert_data = LanguageModelData.from_constructor(bert_data_constructor)
    LanguageModelScorer.display_top_n(bert_data, n=10)

    # SUBSECTION: SciBERT
    scibert_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
    )
    scibert_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=scibert_cosine_similarities_most_cited,
    )
    scibert_data = LanguageModelData.from_constructor(scibert_data_constructor)
    LanguageModelScorer.display_top_n(scibert_data, n=10)

    # SUBSECTION: Longformer
    longformer_cosine_similarities_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.language_models.longformer_cosine_similarities_most_cited_pkl
    )
    longformer_data_constructor = LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=longformer_cosine_similarities_most_cited,
    )
    longformer_data = LanguageModelData.from_constructor(longformer_data_constructor)
    LanguageModelScorer.display_top_n(longformer_data, n=10)

    # SECTION: Evaluate Scores
    average_precision_scores = (
        pd.DataFrame(
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
            columns=["Feature", "Average Precision"],
        )
        .sort_values(by="Average Precision", ascending=False)
        .reset_index(drop=True)
    )

    print(average_precision_scores)


if __name__ == "__main__":
    main()
