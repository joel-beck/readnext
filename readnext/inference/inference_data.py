from dataclasses import dataclass

from typing_extensions import Self

from readnext.inference.constructor import (
    DocumentIdentifier,
    DocumentInfo,
    Features,
    InferenceDataConstructor,
    Labels,
    Ranks,
    Recommendations,
)


@dataclass(kw_only=True)
class InferenceData:
    document_identifier: DocumentIdentifier
    document_info: DocumentInfo
    features: Features
    ranks: Ranks
    labels: Labels
    recommendations: Recommendations

    @classmethod
    def from_constructor(cls, constructor: InferenceDataConstructor) -> Self:
        return cls(
            document_identifier=constructor.collect_document_identifier(),
            document_info=constructor.collect_document_info(),
            features=constructor.collect_features(),
            ranks=constructor.collect_ranks(),
            labels=constructor.collect_labels(),
            recommendations=constructor.collect_recommendations(),
        )
