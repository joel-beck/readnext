from dataclasses import dataclass, field

from readnext.modeling.constructor import ModelDataConstructor
from readnext.utils.aliases import CandidateScoresFrame, LanguageFeaturesFrame, ScoresFrame
from readnext.utils.repr import generate_frame_repr


@dataclass(kw_only=True)
class LanguageModelDataConstructor(ModelDataConstructor):
    """
    Constructor for the language-based recommender model data. Takes the cosine
    similarities of all candidate documents with respect to the query document as
    additional input.
    """

    cosine_similarity_scores_frame: ScoresFrame | CandidateScoresFrame
    info_columns: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "title",
            "author",
            # add publication date as additional info column to inference output if
            # second recommender is language recommender, publication date is a feature
            # column for the citation recommender
            "publication_date",
            "arxiv_labels",
            "semanticscholar_url",
            "arxiv_url",
        ]
    )
    feature_columns: list[str] = field(
        default_factory=lambda: ["candidate_d3_document_id", "cosine_similarity"]
    )

    def __repr__(self) -> str:
        d3_document_id_repr = f"d3_document_id={self.d3_document_id}"
        query_document_repr = f"query_document={self.query_document!r}"
        documents_frame_repr = f"documents_frame={generate_frame_repr(self.documents_frame)}"
        constructor_plugin_repr = f"constructor_plugin={self.constructor_plugin!r}"
        cosine_similarity_scores_frame_repr = (
            f"cosine_similarity_scores_frame="
            f"{generate_frame_repr(self.cosine_similarity_scores_frame)}"
        )
        info_columns_repr = f"info_columns={self.info_columns}"
        feature_columns_repr = f"feature_columns={self.feature_columns}"

        return (
            f"{self.__class__.__name__}(\n"
            f"  {d3_document_id_repr},\n"
            f"  {query_document_repr},\n"
            f"  {documents_frame_repr},\n"
            f"  {constructor_plugin_repr},\n"
            f"  {cosine_similarity_scores_frame_repr},\n"
            f"  {info_columns_repr},\n"
            f"  {feature_columns_repr},\n"
            ")"
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
        Sets the cosine similarity score as the only feature column. The output
        dataframe has two columns named `candidate_d3_document_id` and
        `cosine_similarity`.

        The cosine similarity scores frame contains a subset of the rows of the query
        documents frame. For weighted scoring all candidate documents are kept (left
        join) and the cosine similarity is set to -1 for documents not present in the
        scores frame.
        """

        return (
            self.get_query_documents_frame()
            .join(
                self.get_cosine_similarity_candidate_scores_frame(),
                on="candidate_d3_document_id",
                how="left",
            )
            .rename({"score": "cosine_similarity"})
            .select(self.feature_columns)
            .fill_null(-1)
        )
