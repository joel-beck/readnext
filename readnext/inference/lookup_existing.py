from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import HybridScorer
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.modeling.citation_models import (
    add_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.utils import load_df_from_pickle


@dataclass
class DocumentIdentifiers:
    d3_document_id: int | None = None
    semanticscholar_url: str | None = None
    paper_title: str | None = None


@dataclass
class Features:
    publication_date: pd.Series
    citationcount_document: pd.Series
    citationcount_author: pd.Series
    co_citation_analysis: pd.Series
    bibliographic_coupling: pd.Series
    cosine_similarity: pd.Series


@dataclass
class Ranks:
    publication_date: pd.Series
    citationcount_document: pd.Series
    citationcount_author: pd.Series
    co_citation_analysis: pd.Series
    bibliographic_coupling: pd.Series
    cosine_similarity: pd.DataFrame


@dataclass
class Labels:
    arxiv: pd.Series
    integer: pd.Series


@dataclass(kw_only=True)
class Recommendations:
    citation_to_language_candidates: pd.DataFrame
    citation_to_language: pd.DataFrame
    language_to_citation_candidates: pd.DataFrame
    language_to_citation: pd.DataFrame


@dataclass(kw_only=True)
class DocumentData:
    document_identifiers: DocumentIdentifiers
    document_info: DocumentInfo
    features: Features
    ranks: Ranks
    labels: Labels
    recommendations: Recommendations


class LanguageModelChoice(Enum):
    tfidf = "tfidf"
    bm25 = "bm25"
    word2vec = "word2vec"
    glove = "glove"
    fasttext = "fasttext"
    bert = "bert"
    scibert = "scibert"
    longformer = "longformer"

    def __str__(self) -> str:
        return self.value


def get_cosine_similarities_path(language_model_choice: LanguageModelChoice) -> Path:
    match language_model_choice:
        case LanguageModelChoice.tfidf:
            return ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
        case LanguageModelChoice.bm25:
            return ResultsPaths.language_models.bm25_cosine_similarities_most_cited_pkl
        case LanguageModelChoice.word2vec:
            return ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
        case LanguageModelChoice.glove:
            return ResultsPaths.language_models.glove_cosine_similarities_most_cited_pkl
        case LanguageModelChoice.fasttext:
            return ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
        case LanguageModelChoice.bert:
            return ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl
        case LanguageModelChoice.scibert:
            return ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
        case LanguageModelChoice.longformer:
            return ResultsPaths.language_models.longformer_cosine_similarities_most_cited_pkl
        case _:
            raise ValueError(f"Invalid language model choice: {language_model_choice}")


def main() -> None:
    query_document_id = 206594692
    language_model_choice = LanguageModelChoice.tfidf

    documents_authors_labels_citations_most_cited: pd.DataFrame = load_df_from_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    ).set_index("document_id")
    # NOTE: Remove to evaluate on full data
    documents_authors_labels_citations_most_cited = (
        documents_authors_labels_citations_most_cited.head(1000)
    )

    co_citation_analysis_scores_most_cited: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
    )

    bibliographic_coupling_scores: pd.DataFrame = load_df_from_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl
    )

    cosine_similarities_path = get_cosine_similarities_path(language_model_choice)
    cosine_similarities = load_df_from_pickle(cosine_similarities_path)

    citation_model_data_constructor = CitationModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited.pipe(
            add_feature_rank_cols
        ).pipe(set_missing_publication_dates_to_max_rank),
        co_citation_analysis_scores=co_citation_analysis_scores_most_cited,
        bibliographic_coupling_scores=bibliographic_coupling_scores,
    )
    citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)

    language_model_data_constructor = LanguageModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        cosine_similarities=cosine_similarities,
    )
    language_model_data = LanguageModelData.from_constructor(language_model_data_constructor)

    semanticscholar_url = cast(
        str,
        documents_authors_labels_citations_most_cited.loc[query_document_id, "semanticscholar_url"],
    )

    document_identifiers = DocumentIdentifiers(
        citation_model_data.query_document.document_id,
        semanticscholar_url,
        citation_model_data.query_document.title,
    )

    document_info = citation_model_data.query_document

    features = Features(
        citation_model_data.info_matrix["publication_date"],
        citation_model_data.info_matrix["citationcount_document"],
        citation_model_data.info_matrix["citationcount_author"],
        citation_model_data.info_matrix["co_citation_analysis"],
        citation_model_data.info_matrix["bibliographic_coupling"],
        language_model_data.info_matrix["cosine_similarity"],
    )

    ranks = Ranks(
        citation_model_data.feature_matrix["publication_date_rank"],
        citation_model_data.feature_matrix["citationcount_document_rank"],
        citation_model_data.feature_matrix["citationcount_author_rank"],
        citation_model_data.feature_matrix["co_citation_analysis_rank"],
        citation_model_data.feature_matrix["bibliographic_coupling_rank"],
        language_model_data.cosine_similarity_ranks,
    )

    labels = Labels(
        citation_model_data.info_matrix["arxiv_labels"], citation_model_data.integer_labels
    )

    hybrid_scorer = HybridScorer(
        language_model_name=language_model_choice.value,
        citation_model_data=citation_model_data,
        language_model_data=language_model_data,
    )

    hybrid_scorer.recommend()
    recommendations = Recommendations(
        citation_to_language_candidates=hybrid_scorer.citation_to_language_candidates,
        citation_to_language=hybrid_scorer.citation_to_language_recommendations,
        language_to_citation_candidates=hybrid_scorer.language_to_citation_candidates,
        language_to_citation=hybrid_scorer.language_to_citation_recommendations,
    )

    document_data = DocumentData(
        document_identifiers=document_identifiers,
        document_info=document_info,
        features=features,
        ranks=ranks,
        labels=labels,
        recommendations=recommendations,
    )

    document_data.document_identifiers
    document_data.document_info
    document_data.features
    document_data.ranks
    document_data.labels
    document_data.recommendations

    document_data.recommendations.citation_to_language_candidates
    document_data.recommendations.citation_to_language
    document_data.recommendations.language_to_citation_candidates
    document_data.recommendations.language_to_citation


if __name__ == "__main__":
    main()
