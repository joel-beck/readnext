import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import ScoresFrame

seen_integer_score_dataframes = [
    "co_citation_analysis_scores",
    "bibliographic_coupling_scores",
    "seen_paper_attribute_getter_co_citation_analysis",
    "seen_paper_attribute_getter_bibliographic_coupling",
    "inference_data_constructor_seen_co_citation_analysis_scores",
    "inference_data_constructor_seen_bibliographic_coupling_scores",
]

seen_float_score_dataframes = [
    "tfidf_cosine_similarities",
    "bm25_cosine_similarities",
    "word2vec_cosine_similarities",
    "glove_cosine_similarities",
    "fasttext_cosine_similarities",
    "bert_cosine_similarities",
    "scibert_cosine_similarities",
    "longformer_cosine_similarities",
    "seen_paper_attribute_getter_cosine_similarities_tfidf",
    "seen_paper_attribute_getter_cosine_similarities_bm25",
    "seen_paper_attribute_getter_cosine_similarities_word2vec",
    "seen_paper_attribute_getter_cosine_similarities_glove",
    "seen_paper_attribute_getter_cosine_similarities_fasttext",
    "seen_paper_attribute_getter_cosine_similarities_bert",
    "seen_paper_attribute_getter_cosine_similarities_scibert",
    "seen_paper_attribute_getter_cosine_similarities_longformer",
    "inference_data_constructor_seen_cosine_similarities",
]

seen_score_dataframes = seen_integer_score_dataframes + seen_float_score_dataframes

unseen_integer_score_dataframes = [
    "unseen_paper_attribute_getter_co_citation_analysis",
    "unseen_paper_attribute_getter_bibliographic_coupling",
    "inference_data_constructor_unseen_co_citation_analysis_scores",
    "inference_data_constructor_unseen_bibliographic_coupling_scores",
]

unseen_float_score_dataframes = [
    "unseen_paper_attribute_getter_cosine_similarities_tfidf",
    "unseen_paper_attribute_getter_cosine_similarities_bm25",
    "unseen_paper_attribute_getter_cosine_similarities_word2vec",
    "unseen_paper_attribute_getter_cosine_similarities_glove",
    "unseen_paper_attribute_getter_cosine_similarities_fasttext",
    "unseen_paper_attribute_getter_cosine_similarities_bert",
    "unseen_paper_attribute_getter_cosine_similarities_scibert",
    "unseen_paper_attribute_getter_cosine_similarities_longformer",
    "inference_data_constructor_unseen_cosine_similarities",
]

unseen_score_dataframes = unseen_integer_score_dataframes + unseen_float_score_dataframes

integer_score_dataframes = seen_integer_score_dataframes + unseen_integer_score_dataframes
float_score_dataframes = seen_float_score_dataframes + unseen_float_score_dataframes

score_dataframes = seen_score_dataframes + unseen_score_dataframes


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "score_dataframe",
    lazy_fixture(score_dataframes),
)
def test_score_dataframes(
    score_dataframe: ScoresFrame,
) -> None:
    assert isinstance(score_dataframe, pl.DataFrame)

    # check columns
    assert score_dataframe.shape[1] == 3
    assert score_dataframe.columns == ["query_d3_document_id", "candidate_d3_document_id", "score"]
    # check dtypes of id columns
    assert score_dataframe["query_d3_document_id"].dtype == pl.Int64
    assert score_dataframe["candidate_d3_document_id"].dtype == pl.Int64

    # check scores for first query document
    first_query_document_scores = score_dataframe.filter(
        pl.col("query_d3_document_id") == score_dataframe["query_d3_document_id"][0]
    )
    assert isinstance(first_query_document_scores, pl.DataFrame)
    assert first_query_document_scores.shape[1] == 3
    assert first_query_document_scores.columns == [
        "query_d3_document_id",
        "candidate_d3_document_id",
        "score",
    ]

    # check that document scores are ordered by their score in descending order
    first_document_scores_sorted = first_query_document_scores.sort(by="score", descending=True)
    assert_frame_equal(first_document_scores_sorted, first_query_document_scores)


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "score_dataframe",
    lazy_fixture(seen_score_dataframes),
)
def test_seen_score_dataframes(
    score_dataframe: ScoresFrame,
) -> None:
    # check that scores for all documents in training corpus are present except for the
    # query document
    num_query_documents = score_dataframe["query_d3_document_id"].n_unique()
    num_candidate_documents_per_query = (
        score_dataframe.groupby("query_d3_document_id")
        .agg(pl.n_unique("candidate_d3_document_id"))
        .n_unique()
    )
    assert num_candidate_documents_per_query == num_query_documents - 1


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "score_dataframe",
    lazy_fixture(unseen_score_dataframes),
)
def test_unseen_score_dataframes(
    score_dataframe: ScoresFrame,
) -> None:
    # check that scores dataframe contains only rows with query document id -1 for the
    # unseen query document
    assert score_dataframe["query_d3_document_id"].n_unique() == 1
    assert score_dataframe["query_d3_document_id"].unique().to_list() == [-1]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "score_dataframe",
    lazy_fixture(integer_score_dataframes),
)
def test_integer_score_dataframes(
    score_dataframe: ScoresFrame,
) -> None:
    # check dtype of `score` column
    assert score_dataframe["score"].dtype == pl.Int64


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "score_dataframe",
    lazy_fixture(float_score_dataframes),
)
def test_float_score_dataframes(
    score_dataframe: ScoresFrame,
) -> None:
    # check dtype of `score` column
    assert score_dataframe["score"].dtype == pl.Float64

    # check that all cosine similarity scores are between 0 and 1
    assert score_dataframe["score"].is_between(0, 1).all()
