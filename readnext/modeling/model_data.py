from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd
from typing_extensions import Self

from readnext.modeling.document_info import DocumentInfo
from readnext.modeling.model_data_constructor import (
    CitationModelDataConstructor,
    LanguageModelDataConstructor,
    ModelDataConstructor,
)

T = TypeVar("T", bound=ModelDataConstructor)


@dataclass
class ModelData(ABC, Generic[T]):
    """
    Holds the required data for a recommender model, including information about the
    query document, information features about the candidate documents, and the integer
    labels for the candidate documents.
    """

    query_document: DocumentInfo
    info_matrix: pd.DataFrame
    integer_labels: pd.Series

    @classmethod
    @abstractmethod
    def from_constructor(cls, constructor: T) -> Self:
        """Construct a `ModelData` instance from a `ModelDataConstructor` instance."""

    @abstractmethod
    def __getitem__(self, indices: pd.Index) -> Self:
        """Specify how to index or slice a `ModelData` instance."""


@dataclass
class CitationModelData(ModelData):
    """
    Holds the required data for the citation recommender model. Adds a feature matrix
    which contains the citation and global document features.
    """

    feature_matrix: pd.DataFrame

    @classmethod
    def from_constructor(cls, constructor: CitationModelDataConstructor) -> Self:
        return cls(
            constructor.query_document,
            constructor.get_info_matrix().pipe(constructor.extend_info_matrix),
            constructor.get_integer_labels(),
            constructor.get_feature_matrix(),
        )

    def __getitem__(self, indices: pd.Index) -> Self:
        return self.__class__(
            self.query_document,
            self.info_matrix.loc[indices],
            self.integer_labels.loc[indices],
            self.feature_matrix.loc[indices],
        )


@dataclass
class LanguageModelData(ModelData):
    """
    Holds the required data for the language model recommender model. Adds cosine
    similarity ranks of all candidate documents with respect to the query document.
    """

    cosine_similarity_ranks: pd.DataFrame

    @classmethod
    def from_constructor(cls, constructor: LanguageModelDataConstructor) -> Self:
        return cls(
            constructor.query_document,
            constructor.get_info_matrix().pipe(constructor.extend_info_matrix),
            constructor.get_integer_labels(),
            constructor.get_cosine_similarity_ranks(),
        )

    def __getitem__(self, indices: pd.Index) -> Self:
        return self.__class__(
            self.query_document,
            self.info_matrix.loc[indices],
            self.integer_labels.loc[indices],
            self.cosine_similarity_ranks.loc[indices],
        )
