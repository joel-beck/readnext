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
    query_document: DocumentInfo
    info_matrix: pd.DataFrame
    integer_labels: pd.Series

    @classmethod
    @abstractmethod
    def from_constructor(cls, constructor: T) -> Self:
        ...

    @abstractmethod
    def __getitem__(self, indices: pd.Index) -> Self:
        ...


@dataclass
class CitationModelData(ModelData):
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
    cosine_similarity_ranks: pd.DataFrame

    @classmethod
    def from_constructor(cls, constructor: LanguageModelDataConstructor) -> Self:
        return cls(
            constructor.query_document,
            constructor.get_info_matrix().pipe(constructor.extend_info_matrix),
            constructor.get_integer_labels(),
            constructor.cosine_similarity_ranks(),
        )

    def __getitem__(self, indices: pd.Index) -> Self:
        return self.__class__(
            self.query_document,
            self.info_matrix.loc[indices],
            self.integer_labels.loc[indices],
            self.cosine_similarity_ranks.loc[indices],
        )
