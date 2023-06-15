from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import Self


@dataclass(kw_only=True)
class FeatureWeights:
    """
    Holds the weights for the citation features and global document features for the
    non-language model recommender.
    """

    publication_date: float = Field(default=1.0, ge=0)
    citationcount_document: float = Field(default=1.0, ge=0)
    citationcount_author: float = Field(default=1.0, ge=0)
    co_citation_analysis: float = Field(default=1.0, ge=0)
    bibliographic_coupling: float = Field(default=1.0, ge=0)

    def normalize(self) -> Self:
        """
        Normalize the weights with the L1 norm such that the sum of all weights is 1
        (all weights are non-negative).
        """
        l1_norm = sum(
            [
                self.publication_date,
                self.citationcount_document,
                self.citationcount_author,
                self.co_citation_analysis,
                self.bibliographic_coupling,
            ]
        )

        return self.__class__(
            publication_date=self.publication_date / l1_norm,
            citationcount_document=self.citationcount_document / l1_norm,
            citationcount_author=self.citationcount_author / l1_norm,
            co_citation_analysis=self.co_citation_analysis / l1_norm,
            bibliographic_coupling=self.bibliographic_coupling / l1_norm,
        )
