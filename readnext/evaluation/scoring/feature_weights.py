from dataclasses import dataclass

from typing_extensions import Self


@dataclass
class FeatureWeights:
    """
    Holds the weights for the citation features and global document features for the
    non-language model recommender.
    """

    publication_date: float = 1.0
    citationcount_document: float = 1.0
    citationcount_author: float = 1.0
    co_citation_analysis: float = 1.0
    bibliographic_coupling: float = 1.0

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
