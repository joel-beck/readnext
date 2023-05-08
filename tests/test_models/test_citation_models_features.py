import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from readnext.modeling.citation_models import (
    add_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)


# Test data fixture
@pytest.fixture
def test_data() -> pd.DataFrame:
    data = {
        "publication_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2019-01-01"), None],
        "citationcount_document": [50, 100, 75],
        "citationcount_author": [1000, 2000, 3000],
    }
    return pd.DataFrame(data)


# Test cases for add_feature_rank_cols
def test_add_feature_rank_cols(test_data: pd.DataFrame) -> None:
    expected_data = {
        "publication_date": test_data["publication_date"],
        "citationcount_document": test_data["citationcount_document"],
        "citationcount_author": test_data["citationcount_author"],
        "publication_date_rank": [1.0, 2.0, None],
        "citationcount_document_rank": [3.0, 1.0, 2.0],
        "citationcount_author_rank": [3.0, 2.0, 1.0],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = add_feature_rank_cols(test_data)
    assert_frame_equal(result_df, expected_df)


# Test cases for set_missing_publication_dates_to_max_rank
def test_set_missing_publication_dates_to_max_rank(test_data: pd.DataFrame) -> None:
    input_data = test_data.assign(publication_date_rank=[1.0, 2.0, None])
    expected_data = {
        "publication_date": test_data["publication_date"],
        "citationcount_document": test_data["citationcount_document"],
        "citationcount_author": test_data["citationcount_author"],
        "publication_date_rank": [1.0, 2.0, 3.0],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = set_missing_publication_dates_to_max_rank(input_data)
    assert_frame_equal(result_df, expected_df)


# Edge case: No missing publication dates
def test_set_missing_publication_dates_to_max_rank_no_missing(test_data: pd.DataFrame) -> None:
    input_data = test_data.assign(publication_date_rank=[1.0, 2.0, 3.0])
    result_df = set_missing_publication_dates_to_max_rank(input_data)
    assert_frame_equal(result_df, input_data)


# Test data fixture with more rows and different ranking cases
@pytest.fixture
def extended_test_data() -> pd.DataFrame:
    data = {
        "publication_date": [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2019-01-01"),
            None,
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2017-01-01"),
        ],
        "citationcount_document": [50, 100, 75, 50, 100],
        "citationcount_author": [1000, 2000, 3000, 1000, 2000],
    }
    return pd.DataFrame(data)


# Test cases for add_feature_rank_cols with extended test data
def test_add_feature_rank_cols_extended(extended_test_data: pd.DataFrame) -> None:
    expected_data = {
        "publication_date": extended_test_data["publication_date"],
        "citationcount_document": extended_test_data["citationcount_document"],
        "citationcount_author": extended_test_data["citationcount_author"],
        "publication_date_rank": [1.0, 2.0, None, 3.0, 4.0],
        "citationcount_document_rank": [4.5, 1.5, 3.0, 4.5, 1.5],
        "citationcount_author_rank": [4.5, 2.5, 1.0, 4.5, 2.5],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = add_feature_rank_cols(extended_test_data)
    assert_frame_equal(result_df, expected_df)


# Test cases for set_missing_publication_dates_to_max_rank with extended test data
def test_set_missing_publication_dates_to_max_rank_extended(
    extended_test_data: pd.DataFrame,
) -> None:
    input_data = extended_test_data.assign(publication_date_rank=[1.0, 2.0, None, 3.0, 4.0])
    expected_data = {
        "publication_date": extended_test_data["publication_date"],
        "citationcount_document": extended_test_data["citationcount_document"],
        "citationcount_author": extended_test_data["citationcount_author"],
        "publication_date_rank": [1.0, 2.0, 5.0, 3.0, 4.0],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = set_missing_publication_dates_to_max_rank(input_data)
    assert_frame_equal(result_df, expected_df)


# Edge case: All missing publication dates
def test_set_missing_publication_dates_to_max_rank_all_missing(test_data: pd.DataFrame) -> None:
    input_data = test_data.assign(
        publication_date=[None, None, None], publication_date_rank=[None, None, None]
    )
    expected_data = {
        "publication_date": [None, None, None],
        "citationcount_document": test_data["citationcount_document"],
        "citationcount_author": test_data["citationcount_author"],
        "publication_date_rank": [3, 3, 3],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = set_missing_publication_dates_to_max_rank(input_data)
    assert_frame_equal(result_df, expected_df)
