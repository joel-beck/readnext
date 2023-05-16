from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from readnext.config import ResultsPaths
from readnext.evaluation.scoring import FeatureWeights
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
from readnext.modeling.language_models import (
    LanguageModelChoice,
    load_cosine_similarities_from_choice,
)
from readnext.utils import load_df_from_pickle


@dataclass(kw_only=True)
class DocumentIdentifiers:
    semanticscholar_id: str
    semanticscholar_url: str
    arxiv_id: str
    arxiv_url: str


class NonUniqueError(Exception):
    """Raise when a non-unique document identifier is encountered."""


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

    @abstractmethod
    def get_identifiers(self) -> DocumentIdentifiers:
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
