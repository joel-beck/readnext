import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from pandas.testing import assert_frame_equal

from readnext.config import ResultsPaths
from readnext.modeling import DocumentInfo, DocumentScore
from readnext.utils import load_df_from_pickle


@pytest.fixture(scope="session")
def co_citation_analysis_scores_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
    )


@pytest.fixture(scope="session")
def bibliographic_coupling_scores_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl
    )


@pytest.mark.slow
def test_co_citation_analysis_scores_most_cited(
    co_citation_analysis_scores_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(co_citation_analysis_scores_most_cited, pd.DataFrame)

    # check number and names of columns and index
    assert co_citation_analysis_scores_most_cited.shape[1] == 1
    assert co_citation_analysis_scores_most_cited.index.name == "document_id"
    assert co_citation_analysis_scores_most_cited.columns.tolist() == ["scores"]

    # check document id data type
    assert is_integer_dtype(co_citation_analysis_scores_most_cited.index)

    # check document scores column
    first_document_scores: list[DocumentScore] = co_citation_analysis_scores_most_cited[
        "scores"
    ].iloc[0]
    assert isinstance(first_document_scores, list)

    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(co_citation_analysis_scores_most_cited) - 1)
        for document_scores in co_citation_analysis_scores_most_cited["scores"]
    )

    unique_index_ids = set(co_citation_analysis_scores_most_cited.index.tolist())
    unique_document_ids = {
        document_score.document_info.d3_document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1

    # check data types of document scores
    first_document_score: DocumentScore = first_document_scores[0]
    assert isinstance(first_document_score, DocumentScore)

    first_document_info: DocumentInfo = first_document_score.document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.d3_document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []


@pytest.mark.slow
def test_bibliographic_coupling_scores_most_cited(
    bibliographic_coupling_scores_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(bibliographic_coupling_scores_most_cited, pd.DataFrame)

    # check number and names of columns and index
    assert bibliographic_coupling_scores_most_cited.shape[1] == 1
    assert bibliographic_coupling_scores_most_cited.index.name == "document_id"
    assert bibliographic_coupling_scores_most_cited.columns.tolist() == ["scores"]

    # check document id data type
    assert is_integer_dtype(bibliographic_coupling_scores_most_cited.index)

    # check document scores column
    first_document_scores: list[DocumentScore] = bibliographic_coupling_scores_most_cited[
        "scores"
    ].iloc[0]
    assert isinstance(first_document_scores, list)

    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(bibliographic_coupling_scores_most_cited) - 1)
        for document_scores in bibliographic_coupling_scores_most_cited["scores"]
    )

    unique_index_ids = set(bibliographic_coupling_scores_most_cited.index.tolist())
    unique_document_ids = {
        document_score.document_info.d3_document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1

    # check data types of document scores
    first_document_score: DocumentScore = first_document_scores[0]
    assert isinstance(first_document_score, DocumentScore)

    first_document_info: DocumentInfo = first_document_score.document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.d3_document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []


@pytest.mark.slow
def test_that_test_data_mimics_real_data(
    test_data_size: int,
    co_citation_analysis_scores_most_cited: pd.DataFrame,
    bibliographic_coupling_scores_most_cited: pd.DataFrame,
    test_co_citation_analysis_scores_most_cited: pd.DataFrame,
    test_bibliographic_coupling_scores_most_cited: pd.DataFrame,
) -> None:
    assert_frame_equal(
        co_citation_analysis_scores_most_cited.head(test_data_size),
        test_co_citation_analysis_scores_most_cited,
    )

    assert_frame_equal(
        bibliographic_coupling_scores_most_cited.head(test_data_size),
        test_bibliographic_coupling_scores_most_cited,
    )
