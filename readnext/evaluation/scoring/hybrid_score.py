from dataclasses import dataclass

import polars as pl
from typing_extensions import Self

from readnext.evaluation.scoring.hybrid_scorer import HybridScorer


@dataclass
class HybridScore:
    """
    Holds all scores (e.g. Average Precision) for a hybrid recommender. Makes it easy to
    compare the scores of both recommender component orders as well as the individual
    component scores.
    """

    language_model_name: str
    citation_to_language: float
    citation_to_language_candidates: float
    language_to_citation: float
    language_to_citation_candidates: float

    @classmethod
    def from_scorer(cls, hybrid_scorer: HybridScorer) -> Self:
        """Constructs a HybridScore from a HybridScorer."""
        return cls(
            language_model_name=hybrid_scorer.language_model_name,
            citation_to_language=hybrid_scorer.citation_to_language_scores,
            citation_to_language_candidates=hybrid_scorer.citation_to_language_candidate_scores,
            language_to_citation=hybrid_scorer.language_to_citation_scores,
            language_to_citation_candidates=hybrid_scorer.language_to_citation_candidate_scores,
        )

    def __str__(self) -> str:
        return (
            f"Language Model: {self.language_model_name}\n"
            "---------------------\n"
            f"Citation -> Language\n"
            "---------------------\n"
            f"Candidates Score: {self.citation_to_language_candidates:.3f}\n"
            f"Final Score: {self.citation_to_language:.3f}\n\n"
            f"Language -> Citation\n"
            "---------------------\n"
            f"Candidates Score: {self.language_to_citation_candidates:.3f}\n"
            f"Final Score: {self.language_to_citation:.3f}\n"
        )

    def to_frame(self) -> pl.DataFrame:
        """Collect all scores in a DataFrame."""
        return pl.DataFrame(
            {
                "Language Model": self.language_model_name,
                "Citation -> Language Candidates": round(
                    self.citation_to_language_candidates, ndigits=3
                ),
                "Citation -> Language Final": round(self.citation_to_language, ndigits=3),
                "Language -> Citation Candidates": round(
                    self.language_to_citation_candidates, ndigits=3
                ),
                "Language -> Citation Final": round(self.language_to_citation, ndigits=3),
            },
        )


def compare_hybrid_scores(*hybrid_scores: HybridScore) -> pl.DataFrame:
    """
    Stacks the hybrid recommender scores for multiple query documents vertically in a
    DataFrame.
    """
    return pl.concat([hybrid_score.to_frame() for hybrid_score in hybrid_scores], ignore_index=True)
