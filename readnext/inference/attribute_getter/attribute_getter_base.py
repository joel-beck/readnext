from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference.document_identifier import DocumentIdentifier
from readnext.modeling import CitationModelData, LanguageModelData
from readnext.modeling.language_models import LanguageModelChoice


@dataclass(kw_only=True)
class AttributeGetter(ABC):
    """
    Abstract Base class for getting data attributes in the `InferenceDataConstructor`
    class.
    """

    semanticscholar_id: str | None = None
    semanticscholar_url: str | None = None
    arxiv_id: str | None = None
    arxiv_url: str | None = None
    language_model_choice: LanguageModelChoice
    feature_weights: FeatureWeights
    documents_data: pd.DataFrame

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
    def get_co_citation_analysis_scores(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_bibliographic_coupling_scores(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_cosine_similarities(self) -> pd.DataFrame:
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
