"""Compute evaluation metrics and show best recommendations for a single query document."""

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import CitationModelScorer, FeatureWeights, LanguageModelScorer
from readnext.evaluation.scoring.metrics import AveragePrecisionMetric, CountUniqueLabelsMetric
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


def main() -> None:
    # evaluation for a single query document
    query_document_id = 206594692

    # SECTION: Get Raw Data
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

    average_precision = AveragePrecisionMetric()
    count_unique_labels = CountUniqueLabelsMetric()

    # SECTION: Get Model Data
    # SUBSECTION: Citation Models
    citation_model_data_constructor = CitationModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited.pipe(
            add_feature_rank_cols
        ).pipe(set_missing_publication_dates_to_max_rank),
        co_citation_analysis_scores=co_citation_analysis_scores_most_cited,
        bibliographic_coupling_scores=bibliographic_coupling_scores_most_cited,
    )
    citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)

    print(citation_model_data.query_document)

    CitationModelScorer.display_top_n(citation_model_data, FeatureWeights(), n=10)
    CitationModelScorer.display_top_n(citation_model_data, FeatureWeights(1, 0, 0, 0, 0), n=10)
    CitationModelScorer.display_top_n(citation_model_data, FeatureWeights(0, 1, 0, 0, 0), n=10)
    CitationModelScorer.display_top_n(citation_model_data, FeatureWeights(0, 0, 1, 0, 0), n=10)
    CitationModelScorer.display_top_n(citation_model_data, FeatureWeights(0, 0, 0, 1, 0), n=10)
    CitationModelScorer.display_top_n(citation_model_data, FeatureWeights(0, 0, 0, 0, 1), n=10)

    # SUBSECTION: TF-IDF
    tfidf_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
    )
    tfidf_data_constructor = LanguageModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=tfidf_cosine_similarities_most_cited,
    )
    tfidf_data = LanguageModelData.from_constructor(tfidf_data_constructor)
    LanguageModelScorer.display_top_n(tfidf_data, n=10)

    # SUBSECTION: Word2Vec
    word2vec_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
    )
    word2vec_data_constructor = LanguageModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=word2vec_cosine_similarities_most_cited,
    )
    word2vec_data = LanguageModelData.from_constructor(word2vec_data_constructor)
    LanguageModelScorer.display_top_n(word2vec_data, n=10)

    # SUBSECTION: FastText
    fasttext_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )
    fasttext_data_constructor = LanguageModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=fasttext_cosine_similarities_most_cited,
    )
    fasttext_data = LanguageModelData.from_constructor(fasttext_data_constructor)
    LanguageModelScorer.display_top_n(fasttext_data, n=10)

    # SUBSECTION: BERT
    bert_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl
    )
    bert_data_constructor = LanguageModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=bert_cosine_similarities_most_cited,
    )
    bert_data = LanguageModelData.from_constructor(bert_data_constructor)
    LanguageModelScorer.display_top_n(bert_data, n=10)

    # SUBSECTION: SciBERT
    scibert_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
    )
    scibert_data_constructor = LanguageModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=scibert_cosine_similarities_most_cited,
    )
    scibert_data = LanguageModelData.from_constructor(scibert_data_constructor)
    LanguageModelScorer.display_top_n(scibert_data, n=10)

    # SECTION: Evaluate Scores
    # TODO: Make sure that columns `arxiv_labels` and `integer_labels` are contained in
    # the dataframe!
    average_precision_scores = (
        pd.DataFrame(
            [
                (
                    "Publication Date",
                    CitationModelScorer.score_top_n(
                        citation_model_data, average_precision, FeatureWeights(1, 0, 0, 0, 0), n=20
                    ),
                ),
                (
                    "Citation Count Document",
                    CitationModelScorer.score_top_n(
                        citation_model_data, average_precision, FeatureWeights(0, 1, 0, 0, 0), n=20
                    ),
                ),
                (
                    "Citation Count Author",
                    CitationModelScorer.score_top_n(
                        citation_model_data, average_precision, FeatureWeights(0, 0, 1, 0, 0), n=20
                    ),
                ),
                (
                    "Co-Citation Analysis",
                    CitationModelScorer.score_top_n(
                        citation_model_data, average_precision, FeatureWeights(0, 0, 0, 1, 0), n=20
                    ),
                ),
                (
                    "Bibliographic Coupling",
                    CitationModelScorer.score_top_n(
                        citation_model_data, average_precision, FeatureWeights(0, 0, 0, 0, 1), n=20
                    ),
                ),
                (
                    "Weighted",
                    CitationModelScorer.score_top_n(
                        citation_model_data, average_precision, FeatureWeights(), n=20
                    ),
                ),
                ("TF-IDF", LanguageModelScorer.score_top_n(tfidf_data, average_precision, n=20)),
                (
                    "Word2Vec",
                    LanguageModelScorer.score_top_n(word2vec_data, average_precision, n=20),
                ),
                (
                    "FastText",
                    LanguageModelScorer.score_top_n(fasttext_data, average_precision, n=20),
                ),
                ("BERT", LanguageModelScorer.score_top_n(bert_data, average_precision, n=20)),
                ("SciBERT", LanguageModelScorer.score_top_n(scibert_data, average_precision, n=20)),
            ],
            columns=["Feature", "Average Precision"],
        )
        .sort_values(by="Average Precision", ascending=False)
        .reset_index(drop=True)
    )

    print(average_precision_scores)

    count_unique_labels_scores = (
        pd.DataFrame(
            [
                (
                    "Publication Date",
                    CitationModelScorer.score_top_n(
                        citation_model_data,
                        count_unique_labels,
                        FeatureWeights(1, 0, 0, 0, 0),
                        n=20,
                    ),
                ),
                (
                    "Citation Count Document",
                    CitationModelScorer.score_top_n(
                        citation_model_data,
                        count_unique_labels,
                        FeatureWeights(0, 1, 0, 0, 0),
                        n=20,
                    ),
                ),
                (
                    "Citation Count Author",
                    CitationModelScorer.score_top_n(
                        citation_model_data,
                        count_unique_labels,
                        FeatureWeights(0, 0, 1, 0, 0),
                        n=20,
                    ),
                ),
                (
                    "Co-Citation Analysis",
                    CitationModelScorer.score_top_n(
                        citation_model_data,
                        count_unique_labels,
                        FeatureWeights(0, 0, 0, 1, 0),
                        n=20,
                    ),
                ),
                (
                    "Bibliographic Coupling",
                    CitationModelScorer.score_top_n(
                        citation_model_data,
                        count_unique_labels,
                        FeatureWeights(0, 0, 0, 0, 1),
                        n=20,
                    ),
                ),
                (
                    "Weighted",
                    CitationModelScorer.score_top_n(
                        citation_model_data, count_unique_labels, FeatureWeights(), n=20
                    ),
                ),
                ("TF-IDF", LanguageModelScorer.score_top_n(tfidf_data, count_unique_labels, n=20)),
                (
                    "Word2Vec",
                    LanguageModelScorer.score_top_n(word2vec_data, count_unique_labels, n=20),
                ),
                (
                    "FastText",
                    LanguageModelScorer.score_top_n(fasttext_data, count_unique_labels, n=20),
                ),
                ("BERT", LanguageModelScorer.score_top_n(bert_data, count_unique_labels, n=20)),
                (
                    "SciBERT",
                    LanguageModelScorer.score_top_n(scibert_data, count_unique_labels, n=20),
                ),
            ],
            columns=["Feature", "Average Precision"],
        )
        .sort_values(by="Average Precision", ascending=False)
        .reset_index(drop=True)
    )

    print(count_unique_labels_scores)


if __name__ == "__main__":
    main()