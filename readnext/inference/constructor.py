from dataclasses import field
from typing import Any

from pydantic import ConfigDict, Field, HttpUrl, root_validator, validator
from pydantic.dataclasses import dataclass
from rich import box
from rich.console import Console
from rich.panel import Panel

from readnext.config import DataPaths
from readnext.evaluation.scoring import FeatureWeights, HybridScorer
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
from readnext.inference.features import Features, Labels, Points, Ranks, Recommendations
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils.aliases import DocumentsFrame
from readnext.utils.convert_id_urls import (
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_url_from_semanticscholar_id,
)
from readnext.utils.dummy_defaults import (
    citation_model_data_default,
    documents_frame_default,
    language_model_data_default,
    seen_inference_data_constructor_plugin_default,
)
from readnext.utils.io import read_df_from_parquet
from readnext.utils.repr import generate_frame_repr


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True), kw_only=True)
class InferenceDataConstructor:
    # semanticscholar id is a 40 character hex string
    semanticscholar_id: str | None = Field(
        default=None, min_length=40, max_length=40, regex=r"^[a-fA-F0-9]{40}$"
    )
    semanticscholar_url: HttpUrl | str | None = None
    # arxiv id must start with 4 digits, followed by a dot, followed by 5 digits
    arxiv_id: str | None = Field(
        default=None, min_length=10, max_length=10, regex=r"^\d{4}\.\d{5}$"
    )
    arxiv_url: HttpUrl | str | None = None
    language_model_choice: LanguageModelChoice
    feature_weights: FeatureWeights

    verbose: bool = True

    documents_frame: DocumentsFrame = field(init=False)
    constructor_plugin: InferenceDataConstructorPlugin = field(init=False)
    citation_model_data: CitationModelData = field(init=False)
    language_model_data: LanguageModelData = field(init=False)

    @root_validator(pre=True)
    def check_valid_id_and_url(cls, values: dict[str, Any]) -> dict[str, Any]:
        if (
            values.get("semanticscholar_id") is None
            and values.get("semanticscholar_url") is None
            and values.get("arxiv_id") is None
            and values.get("arxiv_url") is None
        ):
            raise ValueError(
                """
                At least one of semanticscholar_id, semanticscholar_url, arxiv_id,
                arxiv_url must be provided.
                """
            )
        return values

    @validator("semanticscholar_url")
    def validate_semanticscholar_url(cls, semanticscholar_url: str | None) -> str | None:
        if semanticscholar_url is not None and not semanticscholar_url.startswith(
            "https://www.semanticscholar.org/paper/"
        ):
            raise ValueError(
                "Semanticscholar URL must start with `https://www.semanticscholar.org/paper/`"
            )
        return semanticscholar_url

    @validator("arxiv_url")
    def validate_arxiv_url(cls, arxiv_url: str | None) -> str | None:
        if arxiv_url is not None and not arxiv_url.startswith("https://arxiv.org/abs/"):
            raise ValueError("Arxiv URL must start with `https://arxiv.org/abs/`")
        return arxiv_url

    def __post_init__(self) -> None:
        """
        Runs BEFORE pydantic input validation. Sets dummy default values for instance
        attributes that are overwritten by the `__post_init_post_parse__()` method.

        This is necessary for the following reasons:
            - Invalid input arguments for e.g. the `arxiv_url` should be caught by
              pydantic validation
            - Since `__post_init__` runs before validation, the call of
              `self.constructor_plugin.get_citation_model_data()` in `__post_init__()`
              would raise a runtime error for invalid inputs instead of pydantic. Thus
              setting the real instance attribute values is deferred to the
              `__post_init_post_parse__()` method.
            - However, a `__post_init__()` method with a first initialization is still
              necessary to avoid the `field required (type=value_error.missing)` error
              during pydantic validation.
        """
        self.documents_frame = documents_frame_default
        self.constructor_plugin = seen_inference_data_constructor_plugin_default
        self.citation_model_data = citation_model_data_default
        self.language_model_data = language_model_data_default

    def __post_init_post_parse__(self) -> None:
        """
        Runs AFTER pydantic input validation. Overwrites `__post_init__()` instance
        attribute values.

        First, the documents data is loaded to check if the query document is in the
        training data. Based on the result, the model data constructor plugin is set to
        either the one for seen papers or the one for unseen papers. Then, the
        constructor plugin is used to set all data attributes needed for inference.
        """

        # documents data must be set before the constructor plugin since documents data
        # is required to check if the query document is contained in the training data
        # and, thus, to select the correct constructor plugin
        self.documents_frame = self.get_documents_frame()
        self.constructor_plugin = self.get_constructor_plugin()
        self.citation_model_data = self.constructor_plugin.get_citation_model_data()
        self.language_model_data = self.constructor_plugin.get_language_model_data()

    def __repr__(self) -> str:
        semanticscholar_id_repr = f"semanticscholar_id={self.semanticscholar_id}"
        semanticscholar_url_repr = f"semanticscholar_url={self.semanticscholar_url}"
        arxiv_id_repr = f"arxiv_id={self.arxiv_id}"
        arxiv_url_repr = f"arxiv_url={self.arxiv_url}"
        language_model_choice_repr = f"language_model_choice={self.language_model_choice!r}"
        feature_weights_repr = f"feature_weights={self.feature_weights!r}"
        documents_frame_repr = f"documents_frame={generate_frame_repr(self.documents_frame)}"
        constructor_plugin_repr = f"constructor_plugin={self.constructor_plugin!r}"
        citation_model_data_repr = f"citation_model_data={self.citation_model_data!r}"
        language_model_data_repr = f"language_model_data={self.language_model_data!r}"

        return (
            f"{self.__class__.__name__}(\n"
            f"  {semanticscholar_id_repr},\n"
            f"  {semanticscholar_url_repr},\n"
            f"  {arxiv_id_repr},\n"
            f"  {arxiv_url_repr},\n"
            f"  {language_model_choice_repr},\n"
            f"  {feature_weights_repr},\n"
            f"  {documents_frame_repr},\n"
            f"  {constructor_plugin_repr},\n"
            f"  {citation_model_data_repr},\n"
            f"  {language_model_data_repr}\n"
            ")"
        )

    def get_documents_frame(self) -> DocumentsFrame:
        return read_df_from_parquet(DataPaths.merged.documents_frame)

    def query_document_in_training_data(self) -> bool:
        if self.semanticscholar_id is not None:
            semanticscholar_url = get_semanticscholar_url_from_semanticscholar_id(
                self.semanticscholar_id
            )
            return semanticscholar_url in self.documents_frame["semanticscholar_url"].to_list()

        if self.semanticscholar_url is not None:
            return self.semanticscholar_url in self.documents_frame["semanticscholar_url"].to_list()

        if self.arxiv_id is not None:
            return self.arxiv_id in self.documents_frame["arxiv_id"].to_list()

        if self.arxiv_url is not None:
            arxiv_id = get_arxiv_id_from_arxiv_url(self.arxiv_url)
            return arxiv_id in self.documents_frame["arxiv_id"].to_list()

        raise ValueError("No query document identifier provided.")

    def get_constructor_plugin(self) -> InferenceDataConstructorPlugin:
        console = Console()

        if self.query_document_in_training_data():
            # NOTE: Cannot be initialized before calling
            # `self.query_document_in_training_data()` since `post_init()` of
            # `SeenInferenceDataConstructorPlugin` raises an error for unseen documents
            seen_inference_data_constructor_plugin = SeenInferenceDataConstructorPlugin(
                semanticscholar_id=self.semanticscholar_id,
                semanticscholar_url=self.semanticscholar_url,
                arxiv_id=self.arxiv_id,
                arxiv_url=self.arxiv_url,
                language_model_choice=self.language_model_choice,
                feature_weights=self.feature_weights,
                documents_frame=self.documents_frame,
            )

            if not self.verbose:
                return seen_inference_data_constructor_plugin

            console.print(
                Panel.fit(
                    "Query document is contained in the training data",
                    box=box.ROUNDED,
                    padding=(1, 1, 1, 1),
                )
            )

            return seen_inference_data_constructor_plugin

        unseen_inference_data_constructor_plugin = UnseenInferenceDataConstructorPlugin(
            semanticscholar_id=self.semanticscholar_id,
            semanticscholar_url=self.semanticscholar_url,
            arxiv_id=self.arxiv_id,
            arxiv_url=self.arxiv_url,
            language_model_choice=self.language_model_choice,
            feature_weights=self.feature_weights,
            documents_frame=self.documents_frame,
        )

        if not self.verbose:
            return unseen_inference_data_constructor_plugin

        console.print(
            Panel.fit(
                "Query document is [bold]not[/bold] contained in the training data",
                box=box.ROUNDED,
                padding=(1, 1, 1, 1),
            )
        )
        return unseen_inference_data_constructor_plugin

    def collect_document_identifier(self) -> DocumentIdentifier:
        return self.constructor_plugin.identifier

    def collect_document_info(self) -> DocumentInfo:
        return self.citation_model_data.query_document

    def collect_features(self) -> Features:
        return Features(
            publication_date=self.citation_model_data.features_frame.select(
                "candidate_d3_document_id", "publication_date"
            ),
            citationcount_document=self.citation_model_data.features_frame.select(
                "candidate_d3_document_id", "citationcount_document"
            ),
            citationcount_author=self.citation_model_data.features_frame.select(
                "candidate_d3_document_id", "citationcount_author"
            ),
            co_citation_analysis=self.citation_model_data.features_frame.select(
                "candidate_d3_document_id", "co_citation_analysis_score"
            ),
            bibliographic_coupling=self.citation_model_data.features_frame.select(
                "candidate_d3_document_id", "bibliographic_coupling_score"
            ),
            cosine_similarity=self.language_model_data.features_frame.select(
                "candidate_d3_document_id", "cosine_similarity"
            ),
            feature_weights=self.feature_weights,
        )

    def collect_ranks(self) -> Ranks:
        return Ranks(
            publication_date=self.citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "publication_date_rank"
            ),
            citationcount_document=self.citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "citationcount_document_rank"
            ),
            citationcount_author=self.citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "citationcount_author_rank"
            ),
            co_citation_analysis=self.citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "co_citation_analysis_rank"
            ),
            bibliographic_coupling=self.citation_model_data.ranks_frame.select(
                "candidate_d3_document_id", "bibliographic_coupling_rank"
            ),
        )

    def collect_points(self) -> Points:
        return Points(
            publication_date=self.citation_model_data.points_frame.select(
                "candidate_d3_document_id", "publication_date_points"
            ),
            citationcount_document=self.citation_model_data.points_frame.select(
                "candidate_d3_document_id", "citationcount_document_points"
            ),
            citationcount_author=self.citation_model_data.points_frame.select(
                "candidate_d3_document_id", "citationcount_author_points"
            ),
            co_citation_analysis=self.citation_model_data.points_frame.select(
                "candidate_d3_document_id", "co_citation_analysis_points"
            ),
            bibliographic_coupling=self.citation_model_data.points_frame.select(
                "candidate_d3_document_id", "bibliographic_coupling_points"
            ),
        )

    def collect_labels(self) -> Labels:
        return Labels(
            arxiv=self.citation_model_data.info_frame.select(
                "candidate_d3_document_id", "arxiv_labels"
            ),
            integer=self.citation_model_data.integer_labels_frame,
        )

    def collect_recommendations(self) -> Recommendations:
        hybrid_scorer = HybridScorer(
            language_model_name=self.language_model_choice.value,
            citation_model_data=self.citation_model_data,
            language_model_data=self.language_model_data,
        )
        hybrid_scorer.fit(feature_weights=self.feature_weights)

        return Recommendations(
            citation_to_language_candidates=hybrid_scorer.citation_to_language_candidates,
            citation_to_language=hybrid_scorer.citation_to_language_recommendations,
            language_to_citation_candidates=hybrid_scorer.language_to_citation_candidates,
            language_to_citation=hybrid_scorer.language_to_citation_recommendations,
        )
