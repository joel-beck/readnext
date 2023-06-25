import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import CandidateScoresFrame, ScoresFrame

candidate_score_frames = [
    lazy_fixture("citation_model_data_constructor_co_citation_analysis_scores"),
    lazy_fixture("citation_model_data_constructor_bibliographic_coupling_scores"),
]

seen_integer_score_frames = [
    lazy_fixture("test_co_citation_analysis_scores"),
    lazy_fixture("test_bibliographic_coupling_scores"),
]
seen_integer_score_frames_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_co_citation_analysis"),
    lazy_fixture("inference_data_constructor_plugin_seen_bibliographic_coupling"),
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
]
seen_float_score_frames_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_cosine_similarities")
]

unseen_integer_score_frames_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_co_citation_analysis"),
    lazy_fixture("inference_data_constructor_plugin_unseen_bibliographic_coupling"),
]

unseen_float_score_frames_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_cosine_similarities"),
]

seen_score_frames = seen_integer_score_frames + seen_float_score_frames

seen_score_frames_skip_ci = seen_integer_score_frames_skip_ci + seen_float_score_frames_skip_ci

unseen_score_frames_slow_skip_ci = (
    unseen_integer_score_frames_slow_skip_ci + unseen_float_score_frames_slow_skip_ci
)


@pytest.mark.updated
@pytest.mark.parametrize("candidate_scores_frame", candidate_score_frames)
def test_candidate_score_frames(candidate_scores_frame: CandidateScoresFrame) -> None:
    assert isinstance(candidate_scores_frame, pl.DataFrame)

    assert candidate_scores_frame.shape[1] == 2
    assert candidate_scores_frame.columns == ["candidate_d3_document_id", "score"]

    assert candidate_scores_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert candidate_scores_frame["score"].dtype == pl.Int64

    # check that scores are sorted in descending order
    assert candidate_scores_frame["score"].is_sorted(descending=True)


@pytest.mark.updated
@pytest.mark.parametrize(
    "scores_frame",
    [
        *[pytest.param(fixture) for fixture in seen_score_frames],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in seen_score_frames_skip_ci
        ],
    ],
)
def test_seen_score_frames(
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


@pytest.mark.updated
@pytest.mark.parametrize(
    "scores_frame",
    [
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in unseen_score_frames_slow_skip_ci
        ],
    ],
)
def test_unseen_score_frames(
    scores_frame: ScoresFrame,
) -> None:
    assert isinstance(scores_frame, pl.DataFrame)

    assert scores_frame.shape[1] == 2
    assert scores_frame.columns == ["candidate_d3_document_id", "score"]

    assert scores_frame["candidate_d3_document_id"].dtype == pl.Int64

    # check that document scores are ordered by their score in descending order
    assert scores_frame["candidate_d3_document_id"].is_sorted(descending=True)


@pytest.mark.updated
@pytest.mark.parametrize(
    "scores_frame",
    [
        *[pytest.param(fixture) for fixture in seen_integer_score_frames],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in seen_integer_score_frames_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in unseen_integer_score_frames_slow_skip_ci
        ],
    ],
)
def test_integer_score_frames(
    scores_frame: ScoresFrame,
) -> None:
    # check dtype of `score` column
    assert scores_frame["score"].dtype == pl.Int64


@pytest.mark.updated
@pytest.mark.parametrize(
    "scores_frame",
    [
        *[pytest.param(fixture) for fixture in seen_float_score_frames],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in seen_float_score_frames_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in unseen_float_score_frames_slow_skip_ci
        ],
    ],
)
def test_float_score_frames(
    scores_frame: ScoresFrame,
) -> None:
    # check dtype of `score` column
    assert scores_frame["score"].dtype == pl.Float64

    # check that all cosine similarity scores are between 0 and 1
    assert scores_frame["score"].is_between(0, 1).all()
