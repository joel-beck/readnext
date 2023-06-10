from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import polars as pl
from typing_extensions import Self

from readnext.modeling.constructor import ModelDataConstructor
from readnext.modeling.constructor_citation import CitationModelDataConstructor
from readnext.modeling.constructor_language import LanguageModelDataConstructor
from readnext.modeling.document_info import DocumentInfo
from readnext.utils.repr import generate_frame_repr
from readnext.utils.aliases import (
    CitationFeaturesFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    InfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
)

TModelDataConstructor = TypeVar("TModelDataConstructor", bound=ModelDataConstructor)


@dataclass(kw_only=True)
class ModelData(ABC, Generic[TModelDataConstructor]):
    """
    Holds the required data for a recommender model, including information about the
    query document, information features about the candidate documents, and the integer
    labels for the candidate documents.
    """

    query_document: DocumentInfo
    info_frame: InfoFrame
    features_frame: CitationFeaturesFrame | LanguageFeaturesFrame
    integer_labels_frame: IntegerLabelsFrame

    @classmethod
    @abstractmethod
    def from_constructor(cls, model_data_constructor: TModelDataConstructor) -> Self:
        """Construct a `ModelData` instance from a `ModelDataConstructor` instance."""

    @abstractmethod
    def __getitem__(self, indices: list[int]) -> Self:
        """Specify how to index or slice a `ModelData` instance."""

    @abstractmethod
    def __repr__(self) -> str:
        """Specify how to represent a `ModelData` instance as a string."""


@dataclass(kw_only=True)
class CitationModelData(ModelData):
    """
    Holds the required data for the citation recommender model.
    """

    features_frame: CitationFeaturesFrame
    ranks_frame: CitationRanksFrame
    points_frame: CitationPointsFrame

    @classmethod
    def from_constructor(cls, model_data_constructor: CitationModelDataConstructor) -> Self:
        features_frame = model_data_constructor.get_features_frame()
        ranks_frame = model_data_constructor.get_ranks_frame(features_frame)
        points_frame = model_data_constructor.get_points_frame(ranks_frame)

        return cls(
            query_document=model_data_constructor.query_document,
            info_frame=model_data_constructor.get_info_frame(),
            features_frame=features_frame,
            ranks_frame=ranks_frame,
            points_frame=points_frame,
            integer_labels_frame=model_data_constructor.get_integer_labels_frame(),
        )

    def __getitem__(self, indices: list[int]) -> Self:
        return self.__class__(
            query_document=self.query_document,
            info_frame=self.info_frame.filter(pl.col("candidate_d3_document_id").is_in(indices)),
            features_frame=self.features_frame.filter(
                pl.col("candidate_d3_document_id").is_in(indices)
            ),
            ranks_frame=self.ranks_frame.filter(pl.col("candidate_d3_document_id").is_in(indices)),
            points_frame=self.points_frame.filter(
                pl.col("candidate_d3_document_id").is_in(indices)
            ),
            integer_labels_frame=self.integer_labels_frame.filter(
                pl.col("candidate_d3_document_id").is_in(indices)
            ),
        )

    def __repr__(self) -> str:
        query_document_repr = f"query_document={self.query_document!r}"
        info_frame_repr = f"info_frame={generate_frame_repr(self.info_frame)}"
        features_frame_repr = f"features_frame={generate_frame_repr(self.features_frame)}"
        ranks_frame_repr = f"ranks_frame={generate_frame_repr(self.ranks_frame)}"
        points_frame_repr = f"points_frame={generate_frame_repr(self.points_frame)}"
        integer_labels_frame_repr = (
            f"integer_labels_frame={generate_frame_repr(self.integer_labels_frame)}"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  {query_document_repr},\n"
            f"  {info_frame_repr},\n"
            f"  {features_frame_repr},\n"
            f"  {ranks_frame_repr},\n"
            f"  {points_frame_repr},\n"
            f"  {integer_labels_frame_repr}\n"
            ")"
        )


@dataclass(kw_only=True)
class LanguageModelData(ModelData):
    """
    Holds the required data for the language model recommender model. Adds cosine
    similarity ranks of all candidate documents with respect to the query document.
    """

    features_frame: LanguageFeaturesFrame

    @classmethod
    def from_constructor(cls, model_data_constructor: LanguageModelDataConstructor) -> Self:
        return cls(
            query_document=model_data_constructor.query_document,
            info_frame=model_data_constructor.get_info_frame(),
            features_frame=model_data_constructor.get_features_frame(),
            integer_labels_frame=model_data_constructor.get_integer_labels_frame(),
        )

    def __getitem__(self, indices: list[int]) -> Self:
        return self.__class__(
            query_document=self.query_document,
            info_frame=self.info_frame.filter(pl.col("candidate_d3_document_id").is_in(indices)),
            features_frame=self.features_frame.filter(
                pl.col("candidate_d3_document_id").is_in(indices)
            ),
            integer_labels_frame=self.integer_labels_frame.filter(
                pl.col("candidate_d3_document_id").is_in(indices)
            ),
        )

    def __repr__(self) -> str:
        query_document_repr = f"query_document={self.query_document!r}"
        info_frame_repr = f"info_frame={generate_frame_repr(self.info_frame)}"
        features_frame_repr = f"features_frame={generate_frame_repr(self.features_frame)}"
        integer_labels_frame_repr = (
            f"integer_labels_frame={generate_frame_repr(self.integer_labels_frame)}"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  {query_document_repr},\n"
            f"  {info_frame_repr},\n"
            f"  {features_frame_repr},\n"
            f"  {integer_labels_frame_repr}\n"
            ")"
        )
