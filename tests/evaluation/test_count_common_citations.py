import polars as pl

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
    u = pl.Series([1, 2, 3, 4, 5])
    v = pl.Series([4, 5, 6, 7, 8])
    assert CountCommonCitations.score(u, v) == 2


def test_count_common_citations_mixed_inputs() -> None:
    u = [1, 2, 3, 4, 5]
    v = pl.Series([4, 5, 6, 7, 8])
    assert CountCommonCitations.score(u, v) == 2


def test_count_common_citations_from_df() -> None:
    df = pl.DataFrame(
        {
            "d3_document_id": [1, 2, 3],
            "citations": [
                [1, 2, 3, 4, 5],
                [4, 5, 6, 7, 8],
                [1, 3, 5, 7, 9],
            ],
        }
    )
    assert CountCommonCitations.from_df(df, 1, 2) == 2
    assert CountCommonCitations.from_df(df, 1, 3) == 3
    assert CountCommonCitations.from_df(df, 2, 3) == 2


def test_count_common_citations_from_df_different_data_types() -> None:
    df = pl.DataFrame(
        {
            "d3_document_id": [1, 2, 3],
            "citations": [
                ["A", "B", "C", "D", "E"],
                ["D", "E", "F", "G", "H"],
                ["A", "C", "E", "G", "I"],
            ],
        }
    )

    assert CountCommonCitations.from_df(df, 1, 2) == 2
    assert CountCommonCitations.from_df(df, 1, 3) == 3
    assert CountCommonCitations.from_df(df, 2, 3) == 2
