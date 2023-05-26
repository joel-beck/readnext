import pandas as pd
import pytest

from readnext.evaluation.metrics import CountCommonCitations


def test_count_common_citations_basic() -> None:
    u = [1, 2, 3, 4, 5]
    v = [4, 5, 6, 7, 8]
    assert CountCommonCitations.score(u, v) == 2


def test_count_common_citations_identical_vectors() -> None:
    u = [1, 2, 3, 4, 5]
    assert CountCommonCitations.score(u, u) == 5


def test_count_common_citations_no_common_citations() -> None:
    u = [1, 2, 3]
    v = [4, 5, 6]
    assert CountCommonCitations.score(u, v) == 0


def test_count_common_citations_different_lengths() -> None:
    u = [1, 2, 3, 4, 5]
    v = [3, 4, 5, 6]
    assert CountCommonCitations.score(u, v) == 3


def test_count_common_citations_pandas_series() -> None:
    u = pd.Series([1, 2, 3, 4, 5])
    v = pd.Series([4, 5, 6, 7, 8])
    assert CountCommonCitations.score(u, v) == 2


def test_count_common_citations_mixed_inputs() -> None:
    u = [1, 2, 3, 4, 5]
    v = pd.Series([4, 5, 6, 7, 8])
    assert CountCommonCitations.score(u, v) == 2


def test_count_common_citations_from_df() -> None:
    index = pd.Index([1, 2, 3], name="document_id")
    data = {
        "citations": [
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8],
            [1, 3, 5, 7, 9],
        ],
    }
    df = pd.DataFrame(data, index=index)
    assert CountCommonCitations.from_df(df, 1, 2) == 2
    assert CountCommonCitations.from_df(df, 1, 3) == 3
    assert CountCommonCitations.from_df(df, 2, 3) == 2


def test_count_common_citations_from_df_empty_dataframe() -> None:
    index = pd.Index([], name="document_id")
    df = pd.DataFrame(columns=["citations"], index=index)

    with pytest.raises(KeyError):
        CountCommonCitations.from_df(df, 1, 2)


def test_count_common_citations_from_df_missing_document_id() -> None:
    index = pd.Index([1, 2, 3], name="document_id")
    data = {
        "citations": [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8], [1, 3, 5, 7, 9]],
    }
    df = pd.DataFrame(data, index=index)

    with pytest.raises(KeyError):
        CountCommonCitations.from_df(df, 1, 4)


def test_count_common_citations_from_df_different_data_types() -> None:
    index = pd.Index([1, 2, 3], name="document_id")
    data = {
        "citations": [
            ["A", "B", "C", "D", "E"],
            ["D", "E", "F", "G", "H"],
            ["A", "C", "E", "G", "I"],
        ],
    }
    df = pd.DataFrame(data, index=index)

    assert CountCommonCitations.from_df(df, 1, 2) == 2
    assert CountCommonCitations.from_df(df, 1, 3) == 3
    assert CountCommonCitations.from_df(df, 2, 3) == 2
