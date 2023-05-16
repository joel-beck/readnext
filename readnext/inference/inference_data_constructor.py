from dataclasses import dataclass, field

import pandas as pd

from readnext.config import DataPaths
from readnext.evaluation.scoring import FeatureWeights, HybridScorer
from readnext.inference.inference_data_attribute_getter import (
    DocumentIdentifiers,
    InferenceDataAttributeGetter,
    InferenceDataAttributeGetterSeenPaper,
)
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils import load_df_from_pickle


@dataclass
class Features:
    publication_date: pd.Series
    citationcount_document: pd.Series
    citationcount_author: pd.Series
    co_citation_analysis: pd.Series
    bibliographic_coupling: pd.Series
    cosine_similarity: pd.Series
    feature_weights: FeatureWeights


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
class InferenceDataConstructor:
    query_document_id: int | None = None
    semanticscholar_url: str | None = None
    arxiv_url: str | None = None
    paper_title: str | None = None
    language_model_choice: LanguageModelChoice
    feature_weights: FeatureWeights

    attribute_getter: InferenceDataAttributeGetter = field(init=False)

    _documents_data: pd.DataFrame = field(init=False)
    _co_citation_analysis_scores: pd.DataFrame = field(init=False)
    _bibliographic_coupling_scores: pd.DataFrame = field(init=False)
    _cosine_similarities: pd.DataFrame = field(init=False)
    _citation_model_data: CitationModelData = field(init=False)
    _language_model_data: LanguageModelData = field(init=False)

    def __post_init__(self) -> None:
        """
        Collect all information needed for inference right after initialization.

        First, the documents data is loaded to check if the query document is in the
        training data. Based on the result, the attribute getter is set to either the
        one for seen papers or the one for unseen papers. Then, the attribute getter is
        used to set all data attributes needed for inference.
        """

        self._documents_data = self.get_documents_data()
        self.attribute_getter = self.get_attribute_getter()
        self._co_citation_analysis_scores = self.attribute_getter.get_co_citation_analysis_scores()
        self._bibliographic_coupling_scores = (
            self.attribute_getter.get_bibliographic_coupling_scores()
        )
        self._cosine_similarities = self.attribute_getter.get_cosine_similarities()
        self._citation_model_data = self.attribute_getter.get_citation_model_data()
        self._language_model_data = self.attribute_getter.get_language_model_data()

    def get_documents_data(self) -> pd.DataFrame:
        return load_df_from_pickle(
            DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
        ).set_index("document_id")

    # TODO: If the query document id is not provided as an input, it is not yet inferred
    # from e.g. the arxiv url! Change the order of the methods to move initialization
    # the query document id before instantiating the attribute getter.
    def query_document_in_training_data(self) -> bool:
        return self.query_document_id in self._documents_data.index

    def get_attribute_getter(self) -> InferenceDataAttributeGetter:
        if self.query_document_in_training_data():
            return InferenceDataAttributeGetterSeenPaper(
                query_document_id=self.query_document_id,
                semanticscholar_url=self.semanticscholar_url,
                arxiv_url=self.arxiv_url,
                paper_title=self.paper_title,
                language_model_choice=self.language_model_choice,
                feature_weights=self.feature_weights,
                documents_data=self._documents_data,
            )

        raise NotImplementedError

    def collect_document_identifiers(self) -> DocumentIdentifiers:
        return self.attribute_getter.get_identifiers()

    def collect_document_info(self) -> DocumentInfo:
        return self._citation_model_data.query_document

    def collect_features(self) -> Features:
        return Features(
            self._citation_model_data.info_matrix["publication_date"],
            self._citation_model_data.info_matrix["citationcount_document"],
            self._citation_model_data.info_matrix["citationcount_author"],
            self._citation_model_data.info_matrix["co_citation_analysis"],
            self._citation_model_data.info_matrix["bibliographic_coupling"],
            self._language_model_data.info_matrix["cosine_similarity"],
            self.feature_weights,
        )

    def collect_ranks(self) -> Ranks:
        return Ranks(
            self._citation_model_data.feature_matrix["publication_date_rank"],
            self._citation_model_data.feature_matrix["citationcount_document_rank"],
            self._citation_model_data.feature_matrix["citationcount_author_rank"],
            self._citation_model_data.feature_matrix["co_citation_analysis_rank"],
            self._citation_model_data.feature_matrix["bibliographic_coupling_rank"],
            self._language_model_data.cosine_similarity_ranks,
        )

    def collect_labels(self) -> Labels:
        return Labels(
            self._citation_model_data.info_matrix["arxiv_labels"],
            self._citation_model_data.integer_labels,
        )

    def collect_recommendations(self) -> Recommendations:
        hybrid_scorer = HybridScorer(
            language_model_name=self.language_model_choice.value,
            citation_model_data=self._citation_model_data,
            language_model_data=self._language_model_data,
        )
        hybrid_scorer.recommend(feature_weights=self.feature_weights)

        return Recommendations(
            citation_to_language_candidates=hybrid_scorer.citation_to_language_candidates,
            citation_to_language=hybrid_scorer.citation_to_language_recommendations,
            language_to_citation_candidates=hybrid_scorer.language_to_citation_candidates,
            language_to_citation=hybrid_scorer.language_to_citation_recommendations,
        )
