from dataclasses import dataclass, field

from readnext.modeling.constructor import ModelDataConstructor
from readnext.utils import CandidateScoresFrame, LanguageFeaturesFrame, ScoresFrame


@dataclass(kw_only=True)
class LanguageModelDataConstructor(ModelDataConstructor):
    """
    Constructor for the language-based recommender model data. Takes the cosine
    similarities of all candidate documents with respect to the query document as
    additional input.
    """

    cosine_similarity_scores_frame: ScoresFrame | CandidateScoresFrame
    feature_columns: list[str] = field(
        default_factory=lambda: ["candidate_d3_document_id", "cosine_similarity"]
    )

    def get_cosine_similarity_candidate_scores_frame(self) -> CandidateScoresFrame:
        """
        Extracts the cosine similarity scores of all candidate documents with respect to
        the query document and converts them to a dataframe with two columns named
        `candidate_d3_document_id` and `score`.
        """
        return self.constructor_plugin.get_candidate_scores(self.cosine_similarity_scores_frame)

    def get_features_frame(self) -> LanguageFeaturesFrame:
        """
        Sets the cosine similarity score as the only feature column.

        The output dataframe has two columns named `candidate_d3_document_id` and
        `cosine_similarity`.
        """
        return self.get_cosine_similarity_candidate_scores_frame().rename(
            {"score": "cosine_similarity"}
        )
