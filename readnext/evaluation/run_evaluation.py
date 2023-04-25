import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation import CitationModelScorer, LanguageModelScorer, ScoringFeature
from readnext.modeling import CitationModelDataFromId, LanguageModelDataFromId


def main() -> None:
    # evaluation for a single input document
    input_document_id = 206594692

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

    # SECTION: Get Model Data
    citation_model_scorer = CitationModelScorer()
    language_model_scorer = LanguageModelScorer()

    citation_model_data_from_id = CitationModelDataFromId(
        document_id=input_document_id,
        documents_data=documents_authors_labels_citations_most_cited.pipe(
            citation_model_scorer.add_feature_rank_cols
        ).pipe(citation_model_scorer.set_missing_publication_dates_to_max_rank),
        co_citation_analysis_data=co_citation_analysis_most_cited,
        bibliographic_coupling_data=bibliographic_coupling_most_cited,
    )
    citation_model_data = citation_model_data_from_id.get_model_data()

    tfidf_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
    )
    tfidf_data_from_id = LanguageModelDataFromId(
        document_id=input_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarity_matrix=tfidf_cosine_similarities_most_cited,
    )
    tfidf_data = tfidf_data_from_id.get_model_data()

    word2vec_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
    )
    word2vec_data_from_id = LanguageModelDataFromId(
        document_id=input_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarity_matrix=word2vec_cosine_similarities_most_cited,
    )
    word2vec_data = word2vec_data_from_id.get_model_data()

    fasttext_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )
    fasttext_data_from_id = LanguageModelDataFromId(
        document_id=input_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarity_matrix=fasttext_cosine_similarities_most_cited,
    )
    fasttext_data = fasttext_data_from_id.get_model_data()

    # SECTION: Evaluate Scores
    citation_model_scorer.select_top_n_ranks(citation_model_data, ScoringFeature.weighted, n=10)
    citation_model_scorer.score_top_n(citation_model_data, ScoringFeature.weighted, n=10)
    citation_model_scorer.display_top_n(citation_model_data, ScoringFeature.weighted, n=10)

    language_model_scorer.select_top_n_ranks(tfidf_data, n=10)
    language_model_scorer.score_top_n(tfidf_data, n=10)
    language_model_scorer.display_top_n(tfidf_data, n=10)

    language_model_scorer.select_top_n_ranks(word2vec_data, n=10)
    language_model_scorer.score_top_n(word2vec_data, n=10)
    language_model_scorer.display_top_n(word2vec_data, n=10)

    language_model_scorer.select_top_n_ranks(fasttext_data, n=10)
    language_model_scorer.score_top_n(fasttext_data, n=10)
    language_model_scorer.display_top_n(fasttext_data, n=10)


if __name__ == "__main__":
    main()
