import random
from collections.abc import Sequence

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

    @classmethod
    def from_sequence(cls, sequence: Sequence) -> Self:
        return cls(
            publication_date=sequence[0],
            citationcount_document=sequence[1],
            citationcount_author=sequence[2],
            co_citation_analysis=sequence[3],
            bibliographic_coupling=sequence[4],
        )


@dataclass
class FeatureWeightsRanges:
    """
    Defines the ranges for choosing the adequate feature weights during evaluation. Both
    the lower and upper bound are inclusive. The resulting ranges are set as default
    values.
    """

    publication_date: tuple[int, int] = (0, 20)
    citationcount_document: tuple[int, int] = (0, 20)
    citationcount_author: tuple[int, int] = (0, 20)
    co_citation_analysis: tuple[int, int] = (0, 100)
    bibliographic_coupling: tuple[int, int] = (0, 100)

    def sample_one(self) -> tuple[int, int, int, int, int]:
        return (
            random.randint(*self.publication_date),
            random.randint(*self.citationcount_document),
            random.randint(*self.citationcount_author),
            random.randint(*self.co_citation_analysis),
            random.randint(*self.bibliographic_coupling),
        )

    def sample(self, num_samples: int, seed: int) -> list[tuple[int, int, int, int, int]]:
        random.seed(seed)
        return [self.sample_one() for _ in range(num_samples)]
