import pandas as pd
import pytest

from readnext.evaluation.metrics import CountCommonReferences


def test_count_common_references_basic() -> None:
    u = [1, 2, 3, 4, 5]
    v = [4, 5, 6, 7, 8]
    assert CountCommonReferences.score(u, v) == 2


def test_count_common_references_identical_vectors() -> None:
    u = [1, 2, 3, 4, 5]
    assert CountCommonReferences.score(u, u) == 5


def test_count_common_references_no_common_references() -> None:
    u = [1, 2, 3]
    v = [4, 5, 6]
    assert CountCommonReferences.score(u, v) == 0


def test_count_common_references_different_lengths() -> None:
    u = [1, 2, 3, 4, 5]
    v = [3, 4, 5, 6]
    assert CountCommonReferences.score(u, v) == 3


def test_count_common_references_pandas_series() -> None:
    u = pd.Series([1, 2, 3, 4, 5])
    v = pd.Series([4, 5, 6, 7, 8])
    assert CountCommonReferences.score(u, v) == 2


def test_count_common_references_mixed_inputs() -> None:
    u = [1, 2, 3, 4, 5]
    v = pd.Series([4, 5, 6, 7, 8])
    assert CountCommonReferences.score(u, v) == 2


def test_count_common_references_from_df() -> None:
    index = pd.Index([1, 2, 3], name="document_id")
    data = {
        "references": [
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8],
            [1, 3, 5, 7, 9],
        ],
    }
    df = pd.DataFrame(data, index=index)

    assert CountCommonReferences.from_df(df, 1, 2) == 2
    assert CountCommonReferences.from_df(df, 1, 3) == 3
    assert CountCommonReferences.from_df(df, 2, 3) == 2


def test_count_common_references_from_df_empty_dataframe() -> None:
    index = pd.Index([], name="document_id")
    df = pd.DataFrame(columns=["references"], index=index)

    with pytest.raises(KeyError):
        CountCommonReferences.from_df(df, 1, 2)


def test_count_common_references_from_df_missing_document_id() -> None:
    index = pd.Index([1, 2, 3], name="document_id")
    data = {
        "document_id": [1, 2, 3],
        "references": [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [1, 3, 5, 7, 9]],
    }
    df = pd.DataFrame(data, index=index)

    with pytest.raises(KeyError):
        CountCommonReferences.from_df(df, 1, 4)


def test_count_common_references_from_df_different_data_types() -> None:
    index = pd.Index([1, 2, 3], name="document_id")
    data = {
        "document_id": [1, 2, 3],
        "references": [
            ["A", "B", "C", "D", "E"],
            ["D", "E", "F", "G", "H"],
            ["A", "C", "E", "G", "I"],
        ],
    }
    df = pd.DataFrame(data, index=index)

    assert CountCommonReferences.from_df(df, 1, 2) == 2
    assert CountCommonReferences.from_df(df, 1, 3) == 3
    assert CountCommonReferences.from_df(df, 2, 3) == 2
