from dataclasses import dataclass

from typing_extensions import Self

from readnext.inference.inference_data_constructor import (
    DocumentIdentifiers,
    DocumentInfo,
    Features,
    InferenceDataConstructor,
    Labels,
    Ranks,
    Recommendations,
)


@dataclass(kw_only=True)
class InferenceData:
    document_identifiers: DocumentIdentifiers
    document_info: DocumentInfo
    features: Features
    ranks: Ranks
    labels: Labels
    recommendations: Recommendations

    @classmethod
    def from_constructor(cls, constructor: InferenceDataConstructor) -> Self:
        return cls(
            document_identifiers=constructor.collect_document_identifiers(),
            document_info=constructor.collect_document_info(),
            features=constructor.collect_features(),
            ranks=constructor.collect_ranks(),
            labels=constructor.collect_labels(),
            recommendations=constructor.collect_recommendations(),
        )
