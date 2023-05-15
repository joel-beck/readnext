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

TModelDataConstructor = TypeVar("TModelDataConstructor", bound=ModelDataConstructor)


@dataclass
class ModelData(ABC, Generic[TModelDataConstructor]):
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
    def from_constructor(cls, constructor: TModelDataConstructor) -> Self:
        """Construct a `ModelData` instance from a `ModelDataConstructor` instance."""

    @abstractmethod
    def __getitem__(self, indices: pd.Index) -> Self:
        """Specify how to index or slice a `ModelData` instance."""

    @abstractmethod
    def __repr__(self) -> str:
        """Specify how to represent a `ModelData` instance as a string."""


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

    def __repr__(self) -> str:
        query_document_repr = f"query_document={self.query_document!r}"

        info_matrix_repr = (
            f"info_matrix=[pd.DataFrame, shape={self.info_matrix.shape}, "
            f"index={self.info_matrix.index.name}, columns={self.info_matrix.columns.to_list()}]"
        )

        feature_matrix_repr = (
            f"feature_matrix=[pd.DataFrame, shape={self.feature_matrix.shape}, "
            f"index={self.feature_matrix.index.name}, "
            f"columns={self.feature_matrix.columns.to_list()}]"
        )

        integer_labels_repr = (
            f"integer_labels=[pd.Series, shape={self.integer_labels.shape}, "
            f"name={self.integer_labels.name}]"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  {query_document_repr},\n"
            f"  {info_matrix_repr},\n"
            f"  {feature_matrix_repr},\n"
            f"  {integer_labels_repr}\n"
            ")"
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

    def __repr__(self) -> str:
        query_document_repr = f"query_document={self.query_document!r}"

        info_matrix_repr = (
            f"info_matrix=[pd.DataFrame, shape={self.info_matrix.shape}, "
            f"index={self.info_matrix.index.name}, columns={self.info_matrix.columns.to_list()}]"
        )

        cosine_similarity_ranks_repr = (
            f"cosine_similarity_ranks=[pd.DataFrame, shape={self.cosine_similarity_ranks.shape}, "
            f"index={self.cosine_similarity_ranks.index.name}, "
            f"columns={self.cosine_similarity_ranks.columns.to_list()}]"
        )

        integer_labels_repr = (
            f"integer_labels=[pd.Series, shape={self.integer_labels.shape}, "
            f"name={self.integer_labels.name}]"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  {query_document_repr},\n"
            f"  {info_matrix_repr},\n"
            f"  {cosine_similarity_ranks_repr},\n"
            f"  {integer_labels_repr}\n"
            ")"
        )
