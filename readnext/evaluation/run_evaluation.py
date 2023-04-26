import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation import CitationModelScorer, LanguageModelScorer, ScoringFeature
from readnext.modeling import CitationModelDataFromId, LanguageModelDataFromId


def main() -> None:
    # evaluation for a single query document
    query_document_id = 206594692

    # SECTION: Get Raw Data
    documents_authors_labels_citations_most_cited: pd.DataFrame = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    ).set_index("document_id")

    bibliographic_coupling_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_most_cited_pkl
    )

    co_citation_analysis_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.citation_models.co_citation_analysis_most_cited_pkl
    )

    citation_model_scorer = CitationModelScorer()
    language_model_scorer = LanguageModelScorer()

    # SECTION: Get Model Data
    # SUBSECTION: Citation Models
    citation_model_data_from_id = CitationModelDataFromId(
        document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited.pipe(
            citation_model_scorer.add_feature_rank_cols
        ).pipe(citation_model_scorer.set_missing_publication_dates_to_max_rank),
        co_citation_analysis_data=co_citation_analysis_most_cited,
        bibliographic_coupling_data=bibliographic_coupling_most_cited,
    )
    citation_model_data = citation_model_data_from_id.get_model_data()
    citation_model_scorer.display_top_n(citation_model_data, ScoringFeature.weighted, n=10)

    # SUBSECTION: TF-IDF
    tfidf_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
    )
    tfidf_data_from_id = LanguageModelDataFromId(
        document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarity_matrix=tfidf_cosine_similarities_most_cited,
    )
    tfidf_data = tfidf_data_from_id.get_model_data()
    language_model_scorer.display_top_n(tfidf_data, n=10)

    # SUBSECTION: Word2Vec
    word2vec_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
    )
    word2vec_data_from_id = LanguageModelDataFromId(
        document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarity_matrix=word2vec_cosine_similarities_most_cited,
    )
    word2vec_data = word2vec_data_from_id.get_model_data()
    language_model_scorer.display_top_n(word2vec_data, n=10)

    # SUBSECTION: FastText
    fasttext_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )
    fasttext_data_from_id = LanguageModelDataFromId(
        document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarity_matrix=fasttext_cosine_similarities_most_cited,
    )
    fasttext_data = fasttext_data_from_id.get_model_data()
    language_model_scorer.display_top_n(fasttext_data, n=10)

    # SUBSECTION: BERT
    bert_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl
    )
    bert_data_from_id = LanguageModelDataFromId(
        document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarity_matrix=bert_cosine_similarities_most_cited,
    )
    bert_data = bert_data_from_id.get_model_data()
    language_model_scorer.display_top_n(bert_data, n=10)

    # SUBSECTION: SciBERT
    scibert_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
    )
    scibert_data_from_id = LanguageModelDataFromId(
        document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarity_matrix=scibert_cosine_similarities_most_cited,
    )
    scibert_data = scibert_data_from_id.get_model_data()
    language_model_scorer.display_top_n(scibert_data, n=10)

    # SECTION: Evaluate Scores
    pd.DataFrame(
        [
            (
                "Publication Date",
                citation_model_scorer.score_top_n(
                    citation_model_data, ScoringFeature.publication_date, n=20
                ),
            ),
            (
                "Citation Count Document",
                citation_model_scorer.score_top_n(
                    citation_model_data, ScoringFeature.citationcount_document, n=20
                ),
            ),
            (
                "Citation Count Author",
                citation_model_scorer.score_top_n(
                    citation_model_data, ScoringFeature.citationcount_author, n=20
                ),
            ),
            (
                "Co-Citation Analysis",
                citation_model_scorer.score_top_n(
                    citation_model_data, ScoringFeature.co_citation_analysis, n=20
                ),
            ),
            (
                "Bibliographic Coupling",
                citation_model_scorer.score_top_n(
                    citation_model_data, ScoringFeature.bibliographic_coupling, n=20
                ),
            ),
            (
                "Weighted",
                citation_model_scorer.score_top_n(
                    citation_model_data, ScoringFeature.weighted, n=20
                ),
            ),
            ("TF-IDF", language_model_scorer.score_top_n(tfidf_data, n=20)),
            ("Word2Vec", language_model_scorer.score_top_n(word2vec_data, n=20)),
            ("FastText", language_model_scorer.score_top_n(fasttext_data, n=20)),
            ("BERT", language_model_scorer.score_top_n(bert_data, n=20)),
            ("SciBERT", language_model_scorer.score_top_n(scibert_data, n=20)),
        ],
        columns=["Feature", "Average Precision"],
    ).sort_values(by="Average Precision", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    main()
