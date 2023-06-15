import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import ScoresFrame

# TODO: Add score frames from model data constructor plugin
seen_integer_score_frames = [
    lazy_fixture("test_co_citation_analysis_scores"),
    lazy_fixture("test_bibliographic_coupling_scores"),
    # lazy_fixture("seen_paper_attribute_getter_co_citation_analysis"),
    # lazy_fixture("seen_paper_attribute_getter_bibliographic_coupling"),
]


seen_float_score_frames = [
    lazy_fixture("test_tfidf_cosine_similarities"),
    lazy_fixture("test_bm25_cosine_similarities"),
    lazy_fixture("test_word2vec_cosine_similarities"),
    lazy_fixture("test_glove_cosine_similarities"),
    lazy_fixture("test_fasttext_cosine_similarities"),
    lazy_fixture("test_bert_cosine_similarities"),
    lazy_fixture("test_scibert_cosine_similarities"),
    lazy_fixture("test_longformer_cosine_similarities"),
    # lazy_fixture("seen_paper_attribute_getter_cosine_similarities_tfidf"),
    # lazy_fixture("seen_paper_attribute_getter_cosine_similarities_bm25"),
    # lazy_fixture("seen_paper_attribute_getter_cosine_similarities_word2vec"),
    # lazy_fixture("seen_paper_attribute_getter_cosine_similarities_glove"),
    # lazy_fixture("seen_paper_attribute_getter_cosine_similarities_fasttext"),
    # lazy_fixture("seen_paper_attribute_getter_cosine_similarities_bert"),
    # lazy_fixture("seen_paper_attribute_getter_cosine_similarities_scibert"),
    # lazy_fixture("seen_paper_attribute_getter_cosine_similarities_longformer"),
]


# unseen_integer_score_frames = [
# lazy_fixture("unseen_paper_attribute_getter_co_citation_analysis"),
# lazy_fixture("unseen_paper_attribute_getter_bibliographic_coupling"),
# ]

# unseen_float_score_frames = [
# lazy_fixture("unseen_paper_attribute_getter_cosine_similarities_tfidf"),
# lazy_fixture("unseen_paper_attribute_getter_cosine_similarities_bm25"),
# lazy_fixture("unseen_paper_attribute_getter_cosine_similarities_word2vec"),
# lazy_fixture("unseen_paper_attribute_getter_cosine_similarities_glove"),
# lazy_fixture("unseen_paper_attribute_getter_cosine_similarities_fasttext"),
# lazy_fixture("unseen_paper_attribute_getter_cosine_similarities_bert"),
# lazy_fixture("unseen_paper_attribute_getter_cosine_similarities_scibert"),
# lazy_fixture("unseen_paper_attribute_getter_cosine_similarities_longformer"),
# ]

score_frames_fixtures = seen_integer_score_frames + seen_float_score_frames


@pytest.mark.updated
@pytest.mark.parametrize(
    "scores_frame",
    [pytest.param(fixture) for fixture in score_frames_fixtures],
)
def test_score_frames(
    scores_frame: ScoresFrame,
) -> None:
    assert isinstance(scores_frame, pl.DataFrame)

    assert scores_frame.shape[1] == 3
    assert scores_frame.columns == ["query_d3_document_id", "candidate_d3_document_id", "score"]

    assert scores_frame["query_d3_document_id"].dtype == pl.Int64
    assert scores_frame["candidate_d3_document_id"].dtype == pl.Int64

    # get scores for first query document
    query_scores_frame = scores_frame.filter(
        pl.col("query_d3_document_id") == scores_frame["query_d3_document_id"][0]
    )

    # check that document scores are ordered by their score in descending order
    first_document_scores_sorted = query_scores_frame.sort(by="score", descending=True)
    assert_frame_equal(first_document_scores_sorted, query_scores_frame)


# @pytest.mark.updated
# @pytest.mark.parametrize(
#     "scores_frame",
#     # test does not work for testing data since not all query and candidate documents
#     # are contained in the dataset
#     [
#         pytest.param(fixture, marks=(pytest.mark.skip_ci))
#         for fixture in score_frames_fixtures_skip_ci
#     ],
# )
# def test_seen_score_frames(
#     scores_frame: ScoresFrame,
# ) -> None:
#     # check that scores for all documents in training corpus are present except for the
#     # query document
#     num_query_documents = scores_frame["query_d3_document_id"].n_unique()

#     num_candidate_documents_per_query = (
#         scores_frame.groupby("query_d3_document_id")
#         .agg(pl.n_unique("candidate_d3_document_id"))
#         .n_unique()
#     )

#     assert num_candidate_documents_per_query == num_query_documents - 1


# @pytest.mark.updated
# @pytest.mark.parametrize(
#     "scores_frame",
#     [
#         pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
#         for fixture in score_frames_fixtures_slow_skip_ci
#     ],
# )
# def test_unseen_score_frames(
#     scores_frame: ScoresFrame,
# ) -> None:
#     # check that scores dataframe contains only rows with query document id -1 for the
#     # unseen query document
#     assert scores_frame["query_d3_document_id"].n_unique() == 1
#     assert scores_frame["query_d3_document_id"].unique().to_list() == [-1]


@pytest.mark.updated
@pytest.mark.parametrize(
    "scores_frame",
    [pytest.param(fixture) for fixture in seen_integer_score_frames],
)
def test_integer_score_frames(
    scores_frame: ScoresFrame,
) -> None:
    # check dtype of `score` column
    assert scores_frame["score"].dtype == pl.Int64


@pytest.mark.updated
@pytest.mark.parametrize(
    "scores_frame",
    [pytest.param(fixture) for fixture in seen_float_score_frames],
)
def test_float_score_frames(
    scores_frame: ScoresFrame,
) -> None:
    # check dtype of `score` column
    assert scores_frame["score"].dtype == pl.Float64

    # check that all cosine similarity scores are between 0 and 1
    assert scores_frame["score"].is_between(0, 1).all()
