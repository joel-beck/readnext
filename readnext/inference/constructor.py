from dataclasses import dataclass, field

import polars as pl
from rich import box
from rich.console import Console
from rich.panel import Panel

from readnext import FeatureWeights, LanguageModelChoice
from readnext.config import DataPaths
from readnext.evaluation.scoring import HybridScorer
from readnext.inference.constructor_plugin import (
    InferenceDataConstructorPlugin,
)
from readnext.inference.constructor_plugin_seen import (
    SeenInferenceDataConstructorPlugin,
)
from readnext.inference.constructor_plugin_unseen import (
    UnseenInferenceDataConstructorPlugin,
)
from readnext.inference.document_identifier import DocumentIdentifier
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.utils import (
    DocumentsFrame,
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_url_from_semanticscholar_id,
    read_df_from_parquet,
)


@dataclass(kw_only=True)
class Features:
    publication_date: pl.DataFrame
    citationcount_document: pl.DataFrame
    citationcount_author: pl.DataFrame
    co_citation_analysis: pl.DataFrame
    bibliographic_coupling: pl.DataFrame
    cosine_similarity: pl.DataFrame
    feature_weights: FeatureWeights


@dataclass(kw_only=True)
class Ranks:
    publication_date: pl.DataFrame
    citationcount_document: pl.DataFrame
    citationcount_author: pl.DataFrame
    co_citation_analysis: pl.DataFrame
    bibliographic_coupling: pl.DataFrame


@dataclass(kw_only=True)
class Points:
    publication_date: pl.DataFrame
    citationcount_document: pl.DataFrame
    citationcount_author: pl.DataFrame
    co_citation_analysis: pl.DataFrame
    bibliographic_coupling: pl.DataFrame


@dataclass(kw_only=True)
class Labels:
    arxiv: pl.DataFrame
    integer: pl.DataFrame


@dataclass(kw_only=True)
class Recommendations:
    citation_to_language_candidates: pl.DataFrame
    citation_to_language: pl.DataFrame
    language_to_citation_candidates: pl.DataFrame
    language_to_citation: pl.DataFrame


@dataclass(kw_only=True)
class InferenceDataConstructor:
    semanticscholar_id: str | None = None
    semanticscholar_url: str | None = None
    arxiv_id: str | None = None
    arxiv_url: str | None = None
    language_model_choice: LanguageModelChoice
    feature_weights: FeatureWeights

    constructor_plugin: InferenceDataConstructorPlugin = field(init=False)

    _documents_frame: DocumentsFrame = field(init=False)
    _citation_model_data: CitationModelData = field(init=False)
    _language_model_data: LanguageModelData = field(init=False)

    def __post_init__(self) -> None:
        """
        Collect all information needed for inference right after initialization.

        First, the documents data is loaded to check if the query document is in the
        training data. Based on the result, the model data constructor plugin is set to
        either the one for seen papers or the one for unseen papers. Then, the
        constructor plugin is used to set all data attributes needed for inference.
        """

        # documents data must be set before the constructor plugin since documents data
        # is required to check if the query document is contained in the training data
        # and, thus, to select the correct constructor plugin
        self._documents_frame = self.get_documents_frame()
        self.constructor_plugin = self.get_constructor_plugin()
        self._citation_model_data = self.constructor_plugin.get_citation_model_data()
        self._language_model_data = self.constructor_plugin.get_language_model_data()

    def get_documents_frame(self) -> DocumentsFrame:
        return read_df_from_parquet(DataPaths.merged.documents_frame)

    def query_document_in_training_data(self) -> bool:
        if self.semanticscholar_id is not None:
            semanticscholar_url = get_semanticscholar_url_from_semanticscholar_id(
                self.semanticscholar_id
            )
            return semanticscholar_url in self._documents_frame["semanticscholar_url"].to_list()

        if self.semanticscholar_url is not None:
            return (
                self.semanticscholar_url in self._documents_frame["semanticscholar_url"].to_list()
            )

        if self.arxiv_id is not None:
            return self.arxiv_id in self._documents_frame["arxiv_id"].to_list()

        if self.arxiv_url is not None:
            arxiv_id = get_arxiv_id_from_arxiv_url(self.arxiv_url)
            return arxiv_id in self._documents_frame["arxiv_id"].to_list()

        raise ValueError("No query document identifier provided.")

    def get_constructor_plugin(self) -> InferenceDataConstructorPlugin:
        console = Console()

        if self.query_document_in_training_data():
            console.print(
                Panel.fit(
                    "Query document is contained in the training data",
                    box=box.ROUNDED,
                    padding=(1, 1, 1, 1),
                )
            )

            return SeenInferenceDataConstructorPlugin(
                semanticscholar_id=self.semanticscholar_id,
                semanticscholar_url=self.semanticscholar_url,
                arxiv_id=self.arxiv_id,
                arxiv_url=self.arxiv_url,
                language_model_choice=self.language_model_choice,
                feature_weights=self.feature_weights,
                documents_frame=self._documents_frame,
            )

        console.print(
            Panel.fit(
                "Query document is [bold]not[/bold] contained in the training data",
                box=box.ROUNDED,
                padding=(1, 1, 1, 1),
            )
        )
        return UnseenInferenceDataConstructorPlugin(
            semanticscholar_id=self.semanticscholar_id,
            semanticscholar_url=self.semanticscholar_url,
            arxiv_id=self.arxiv_id,
            arxiv_url=self.arxiv_url,
            language_model_choice=self.language_model_choice,
            feature_weights=self.feature_weights,
            documents_frame=self._documents_frame,
        )

    def collect_document_identifier(self) -> DocumentIdentifier:
        return self.constructor_plugin.identifier

    def collect_document_info(self) -> DocumentInfo:
        return self._citation_model_data.query_document

    def collect_features(self) -> Features:
        return Features(
            publication_date=self._citation_model_data.features_frame.select(
                "candidate_d3_document_id", "publication_date"
            ),
            citationcount_document=self._citation_model_data.features_frame.select(
                "candidate_d3_document_id", "citationcount_document"
            ),
            citationcount_author=self._citation_model_data.features_frame.select(
                "candidate_d3_document_id", "citationcount_author"
            ),
            co_citation_analysis=self._citation_model_data.features_frame.select(
                "candidate_d3_document_id", "co_citation_analysis_score"
            ),
            bibliographic_coupling=self._citation_model_data.features_frame.select(
                "candidate_d3_document_id", "bibliographic_coupling_score"
            ),
            cosine_similarity=self._language_model_data.features_frame.select(
                "candidate_d3_document_id", "cosine_similarity"
            ),
            feature_weights=self.feature_weights,
        )

    def collect_ranks(self) -> Ranks:
        return Ranks(
            publication_date=self._citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "publication_date_rank"
            ),
            citationcount_document=self._citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "citationcount_document_rank"
            ),
            citationcount_author=self._citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "citationcount_author_rank"
            ),
            co_citation_analysis=self._citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "co_citation_analysis_rank"
            ),
            bibliographic_coupling=self._citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "bibliographic_coupling_rank"
            ),
        )

    def collect_points(self) -> Points:
        return Points(
            publication_date=self._citation_model_data.points_frame.select(
                "candidate_d3_document_id", "publication_date_points"
            ),
            citationcount_document=self._citation_model_data.points_frame.select(
                "candidate_d3_document_id", "citationcount_document_points"
            ),
            citationcount_author=self._citation_model_data.points_frame.select(
                "candidate_d3_document_id", "citationcount_author_points"
            ),
            co_citation_analysis=self._citation_model_data.points_frame.select(
                "candidate_d3_document_id", "co_citation_analysis_points"
            ),
            bibliographic_coupling=self._citation_model_data.points_frame.select(
                "candidate_d3_document_id", "bibliographic_coupling_points"
            ),
        )

    def collect_labels(self) -> Labels:
        return Labels(
            arxiv=self._citation_model_data.info_frame.select(
                "candidate_d3_document_id", "arxiv_labels"
            ),
            integer=self._citation_model_data.integer_labels_frame,
        )

    def collect_recommendations(self) -> Recommendations:
        hybrid_scorer = HybridScorer(
            language_model_name=self.language_model_choice.value,
            citation_model_data=self._citation_model_data,
            language_model_data=self._language_model_data,
        )
        hybrid_scorer.fit(feature_weights=self.feature_weights)

        return Recommendations(
            citation_to_language_candidates=hybrid_scorer.citation_to_language_candidates,
            citation_to_language=hybrid_scorer.citation_to_language_recommendations,
            language_to_citation_candidates=hybrid_scorer.language_to_citation_candidates,
            language_to_citation=hybrid_scorer.language_to_citation_recommendations,
        )
