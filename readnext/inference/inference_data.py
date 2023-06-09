from dataclasses import dataclass

from typing_extensions import Self

from readnext.inference.constructor import (
    DocumentIdentifier,
    DocumentInfo,
    InferenceDataConstructor,
)
from readnext.inference.features import (
    Features,
    Labels,
    Points,
    Ranks,
    Recommendations,
)


@dataclass(kw_only=True)
class InferenceData:
    document_identifier: DocumentIdentifier
    document_info: DocumentInfo
    features: Features
    ranks: Ranks
    points: Points
    labels: Labels
    recommendations: Recommendations

    def __repr__(self) -> str:
        document_identifier_repr = f"document_identifier={self.document_identifier!r}"
        document_info_repr = f"document_info={self.document_info!r}"
        features_repr = f"features={self.features}"
        ranks_repr = f"ranks={self.ranks}"
        points_repr = f"points={self.points!r}"
        labels_repr = f"labels={self.labels!r}"
        recommendations_repr = f"recommendations={self.recommendations!r}"

        return (
            f"{self.__class__.__name__}(\n"
            f"  {document_identifier_repr},\n"
            f"  {document_info_repr},\n"
            f"  {features_repr},\n"
            f"  {ranks_repr},\n"
            f"  {points_repr},\n"
            f"  {labels_repr},\n"
            f"  {recommendations_repr}\n"
            ")"
        )

    @classmethod
    def from_constructor(cls, constructor: InferenceDataConstructor) -> Self:
        return cls(
            document_identifier=constructor.collect_document_identifier(),
            document_info=constructor.collect_document_info(),
            features=constructor.collect_features(),
            ranks=constructor.collect_ranks(),
            points=constructor.collect_points(),
            labels=constructor.collect_labels(),
            recommendations=constructor.collect_recommendations(),
        )
