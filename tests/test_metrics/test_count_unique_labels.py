import pandas as pd
import pytest

from readnext.evaluation.metrics import CountUniqueLabels


def test_count_unique_labels_empty_list() -> None:
    label_list: list[str] = []
    assert CountUniqueLabels.score(label_list) == 0


def test_count_unique_labels_single_list() -> None:
    label_list = [["a", "b", "c"]]
    assert CountUniqueLabels.score(label_list) == 3


def test_count_unique_labels_multiple_lists() -> None:
    label_list = [["a", "b", "c"], ["a", "d", "e"]]
    assert CountUniqueLabels.score(label_list) == 5


def test_count_unique_labels_with_duplicates() -> None:
    label_list = [["a", "b", "a"], ["a", "a", "b", "c"]]
    assert CountUniqueLabels.score(label_list) == 3


@pytest.mark.parametrize(
    ("input_list", "expected_output"),
    [
        (["a", "b", "c"], 3),
        (["a", "a", "b"], 2),
        (["a"], 1),
        ([], 0),
    ],
)
def test_count_unique_labels_single_list_parametrized(
    input_list: list[str], expected_output: int
) -> None:
    label_list = [input_list]
    assert CountUniqueLabels.score(label_list) == expected_output


def test_count_unique_labels_tuple_of_tuples() -> None:
    label_list = (("a", "b", "c"), ("a", "d", "e"))
    assert CountUniqueLabels.score(label_list) == 5


def test_count_unique_labels_tuple_of_lists() -> None:
    label_list = (["a", "b", "c"], ["a", "d", "e"])
    assert CountUniqueLabels.score(label_list) == 5


def test_count_unique_labels_list_of_tuples() -> None:
    label_list = [("a", "b", "c"), ("a", "d", "e")]
    assert CountUniqueLabels.score(label_list) == 5


def test_count_unique_labels_series_of_lists() -> None:
    data = [
        ["a", "b", "c"],
        ["a", "d", "e"],
        ["a", "b", "a"],
        ["a", "a", "b", "c"],
    ]
    series = pd.Series(data)
    assert CountUniqueLabels.score(series) == 5


def test_count_unique_labels_series_of_tuples() -> None:
    data = [
        ("a", "b", "c"),
        ("a", "d", "e"),
        ("a", "b", "a"),
        ("a", "a", "b", "c"),
    ]
    series = pd.Series(data)
    assert CountUniqueLabels.score(series) == 5


@pytest.mark.parametrize(
    ("input_list", "expected_output"),
    [
        (("a", "b", "c"), 3),
        (("a", "a", "b"), 2),
        (("a",), 1),
        ((), 0),
    ],
)
def test_count_unique_labels_single_tuple_parametrized(
    input_list: list[str], expected_output: int
) -> None:
    label_list = [input_list]
    assert CountUniqueLabels.score(label_list) == expected_output


def test_count_unique_labels_from_df() -> None:
    data = {
        "arxiv_labels": [
            ["a", "b", "c"],
            ["a", "d", "e"],
            ["a", "b", "a"],
            ["a", "a", "b", "c"],
        ]
    }
    df = pd.DataFrame(data)
    assert CountUniqueLabels.from_df(df) == 5


def test_count_unique_labels_from_df_empty() -> None:
    data: dict[str, list[int]] = {"arxiv_labels": []}
    df = pd.DataFrame(data)
    assert CountUniqueLabels.from_df(df) == 0


def test_count_unique_labels_from_df_tuples() -> None:
    data = {
        "arxiv_labels": [
            ("a", "b", "c"),
            ("a", "d", "e"),
            ("a", "b", "a"),
            ("a", "a", "b", "c"),
        ]
    }
    df = pd.DataFrame(data)
    assert CountUniqueLabels.from_df(df) == 5


def test_count_unique_labels_from_df_mixed_lists_and_tuples() -> None:
    data = {
        "arxiv_labels": [
            ["a", "b", "c"],
            ("a", "d", "e"),
            ["a", "b", "a"],
            ("a", "a", "b", "c"),
        ]
    }
    df = pd.DataFrame(data)
    assert CountUniqueLabels.from_df(df) == 5


@pytest.mark.parametrize(
    ("arxiv_labels_column", "expected_output"),
    [
        ([["a", "b", "c"], ["a", "d", "e"], ["a", "b", "a"], ["a", "a", "b", "c"]], 5),
        ([("a", "b", "c"), ("a", "d", "e"), ("a", "b", "a"), ("a", "a", "b", "c")], 5),
        ([["a", "b", "c"], ("a", "d", "e"), ["a", "b", "a"], ("a", "a", "b", "c")], 5),
    ],
)
def test_count_unique_labels_from_df_parametrized(
    arxiv_labels_column: list[list[str]], expected_output: int
) -> None:
    data = {"arxiv_labels": arxiv_labels_column}
    df = pd.DataFrame(data)
    assert CountUniqueLabels.from_df(df) == expected_output
