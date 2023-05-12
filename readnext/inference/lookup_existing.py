from dataclasses import dataclass
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

documents_authors_labels_citations_most_cited: pd.DataFrame = pd.read_pickle(
    DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
).set_index("document_id")
# NOTE: Remove to evaluate on full data
documents_authors_labels_citations_most_cited = documents_authors_labels_citations_most_cited.head(
    1000
)

query_document_id = cast(int, documents_authors_labels_citations_most_cited.index[0])

co_citation_analysis_scores_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
)

bibliographic_coupling_scores_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl
)


citation_model_data_constructor = CitationModelDataConstructor(
    query_document_id=query_document_id,
    documents_data=documents_authors_labels_citations_most_cited.pipe(add_feature_rank_cols).pipe(
        set_missing_publication_dates_to_max_rank
    ),
    co_citation_analysis_scores=co_citation_analysis_scores_most_cited,
    bibliographic_coupling_scores=bibliographic_coupling_scores_most_cited,
)
citation_model_data = CitationModelData.from_constructor(citation_model_data_constructor)


tfidf_cosine_similarities_most_cited: pd.DataFrame = pd.read_pickle(
    ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
)
tfidf_data_constructor = LanguageModelDataConstructor(
    query_document_id=query_document_id,
    documents_data=documents_authors_labels_citations_most_cited,
    cosine_similarities=tfidf_cosine_similarities_most_cited,
)
tfidf_data = LanguageModelData.from_constructor(tfidf_data_constructor)


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


semanticscholar_url = cast(
    str, documents_authors_labels_citations_most_cited.loc[query_document_id, "semanticscholar_url"]
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
    tfidf_data.info_matrix["cosine_similarity"],
)

ranks = Ranks(
    citation_model_data.feature_matrix["publication_date_rank"],
    citation_model_data.feature_matrix["citationcount_document_rank"],
    citation_model_data.feature_matrix["citationcount_author_rank"],
    citation_model_data.feature_matrix["co_citation_analysis_rank"],
    citation_model_data.feature_matrix["bibliographic_coupling_rank"],
    tfidf_data.cosine_similarity_ranks,
)

labels = Labels(citation_model_data.info_matrix["arxiv_labels"], citation_model_data.integer_labels)

tfidf_hybrid_scorer = HybridScorer(
    language_model_name="Tf-Idf",
    citation_model_data=citation_model_data,
    language_model_data=tfidf_data,
)

tfidf_hybrid_scorer.recommend()

recommendations = Recommendations(
    citation_to_language_candidates=tfidf_hybrid_scorer.citation_to_language_candidates,
    citation_to_language=tfidf_hybrid_scorer.citation_to_language_recommendations,
    language_to_citation_candidates=tfidf_hybrid_scorer.language_to_citation_candidates,
    language_to_citation=tfidf_hybrid_scorer.language_to_citation_recommendations,
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
