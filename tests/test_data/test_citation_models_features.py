import pandas as pd
from pandas.testing import assert_frame_equal

from readnext.data import (
    add_citation_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)


def test_add_feature_rank_cols(citation_models_features_frame: pd.DataFrame) -> None:
    expected_data = {
        "publication_date": citation_models_features_frame["publication_date"],
        "citationcount_document": citation_models_features_frame["citationcount_document"],
        "citationcount_author": citation_models_features_frame["citationcount_author"],
        "publication_date_rank": [1.0, 2.0, None],
        "citationcount_document_rank": [3.0, 1.0, 2.0],
        "citationcount_author_rank": [3.0, 2.0, 1.0],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = add_citation_feature_rank_cols(citation_models_features_frame)
    assert_frame_equal(result_df, expected_df)


def test_set_missing_publication_dates_to_max_rank(
    citation_models_features_frame: pd.DataFrame,
) -> None:
    input_data = citation_models_features_frame.assign(publication_date_rank=[1.0, 2.0, None])
    expected_data = {
        "publication_date": citation_models_features_frame["publication_date"],
        "citationcount_document": citation_models_features_frame["citationcount_document"],
        "citationcount_author": citation_models_features_frame["citationcount_author"],
        "publication_date_rank": [1.0, 2.0, 3.0],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = set_missing_publication_dates_to_max_rank(input_data)
    assert_frame_equal(result_df, expected_df)


def test_set_missing_publication_dates_to_max_rank_no_missing(
    citation_models_features_frame: pd.DataFrame,
) -> None:
    input_data = citation_models_features_frame.assign(publication_date_rank=[1.0, 2.0, 3.0])
    result_df = set_missing_publication_dates_to_max_rank(input_data)
    assert_frame_equal(result_df, input_data)


def test_add_feature_rank_cols_extended(
    extended_citation_models_features_frame: pd.DataFrame,
) -> None:
    expected_data = {
        "publication_date": extended_citation_models_features_frame["publication_date"],
        "citationcount_document": extended_citation_models_features_frame["citationcount_document"],
        "citationcount_author": extended_citation_models_features_frame["citationcount_author"],
        "publication_date_rank": [1.0, 2.0, None, 3.0, 4.0],
        "citationcount_document_rank": [4.5, 1.5, 3.0, 4.5, 1.5],
        "citationcount_author_rank": [4.5, 2.5, 1.0, 4.5, 2.5],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = add_citation_feature_rank_cols(extended_citation_models_features_frame)
    assert_frame_equal(result_df, expected_df)


def test_set_missing_publication_dates_to_max_rank_extended(
    extended_citation_models_features_frame: pd.DataFrame,
) -> None:
    input_data = extended_citation_models_features_frame.assign(
        publication_date_rank=[1.0, 2.0, None, 3.0, 4.0]
    )
    expected_data = {
        "publication_date": extended_citation_models_features_frame["publication_date"],
        "citationcount_document": extended_citation_models_features_frame["citationcount_document"],
        "citationcount_author": extended_citation_models_features_frame["citationcount_author"],
        "publication_date_rank": [1.0, 2.0, 5.0, 3.0, 4.0],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = set_missing_publication_dates_to_max_rank(input_data)
    assert_frame_equal(result_df, expected_df)


def test_set_missing_publication_dates_to_max_rank_all_missing(
    citation_models_features_frame: pd.DataFrame,
) -> None:
    input_data = citation_models_features_frame.assign(
        publication_date=[None, None, None], publication_date_rank=[None, None, None]
    )
    expected_data = {
        "publication_date": [None, None, None],
        "citationcount_document": citation_models_features_frame["citationcount_document"],
        "citationcount_author": citation_models_features_frame["citationcount_author"],
        "publication_date_rank": [3, 3, 3],
    }
    expected_df = pd.DataFrame(expected_data)
    result_df = set_missing_publication_dates_to_max_rank(input_data)
    assert_frame_equal(result_df, expected_df)
