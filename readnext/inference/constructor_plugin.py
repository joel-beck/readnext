from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import polars as pl

from readnext import FeatureWeights, LanguageModelChoice
from readnext.inference.document_identifier import DocumentIdentifier
from readnext.modeling import CitationModelData, LanguageModelData
from readnext.utils import DocumentsFrame


@dataclass(kw_only=True)
class InferenceDataConstructorPlugin(ABC):
    """
    Provides methods for the `InferenceDataConstructor` class that are different for
    seen and unseen papers.
    """

    semanticscholar_id: str | None = None
    semanticscholar_url: str | None = None
    arxiv_id: str | None = None
    arxiv_url: str | None = None
    language_model_choice: LanguageModelChoice
    feature_weights: FeatureWeights
    documents_frame: DocumentsFrame

    identifier: DocumentIdentifier = field(init=False)

    def __post_init__(self) -> None:
        self.identifier = self.get_identifier()

    @abstractmethod
    def get_identifier_from_semanticscholar_id(self, semanticscholar_id: str) -> DocumentIdentifier:
        ...

    @abstractmethod
    def get_identifier_from_semanticscholar_url(
        self, semanticscholar_url: str
    ) -> DocumentIdentifier:
        ...

    @abstractmethod
    def get_identifier_from_arxiv_id(self, arxiv_id: str) -> DocumentIdentifier:
        ...

    @abstractmethod
    def get_identifier_from_arxiv_url(self, arxiv_url: str) -> DocumentIdentifier:
        ...

    @abstractmethod
    def get_co_citation_analysis_scores(self) -> pl.DataFrame:
        ...

    @abstractmethod
    def get_bibliographic_coupling_scores(self) -> pl.DataFrame:
        ...

    @abstractmethod
    def get_cosine_similarities(self) -> pl.DataFrame:
        ...

    @abstractmethod
    def get_citation_model_data(self) -> CitationModelData:
        ...

    @abstractmethod
    def get_language_model_data(self) -> LanguageModelData:
        ...

    def get_identifier(self) -> DocumentIdentifier:
        if self.semanticscholar_id is not None:
            return self.get_identifier_from_semanticscholar_id(self.semanticscholar_id)

        if self.semanticscholar_url is not None:
            return self.get_identifier_from_semanticscholar_url(self.semanticscholar_url)

        if self.arxiv_id is not None:
            return self.get_identifier_from_arxiv_id(self.arxiv_id)

        if self.arxiv_url is not None:
            return self.get_identifier_from_arxiv_url(self.arxiv_url)

        raise ValueError("No identifier provided.")
