from readnext.utils import slice_mapping


def test_slice_mapping_empty() -> None:
    empty_dict: dict = {}
    result = slice_mapping(empty_dict)
    assert result == {}


def test_slice_mapping_full() -> None:
    mapping = {"a": 1, "b": 2, "c": 3}
    result = slice_mapping(mapping)
    assert result == mapping


def test_slice_mapping_size() -> None:
    mapping = {"a": 1, "b": 2, "c": 3}
    result = slice_mapping(mapping, size=2)
    expected_result = {"a": 1, "b": 2}
    assert result == expected_result


def test_slice_mapping_start() -> None:
    mapping = {"a": 1, "b": 2, "c": 3}
    result = slice_mapping(mapping, start=1)
    expected_result = {"b": 2, "c": 3}
    assert result == expected_result


def test_slice_mapping_end() -> None:
    mapping = {"a": 1, "b": 2, "c": 3}
    result = slice_mapping(mapping, end=2)
    expected_result = {"a": 1, "b": 2}
    assert result == expected_result


def test_slice_mapping_start_end() -> None:
    mapping = {"a": 1, "b": 2, "c": 3}
    result = slice_mapping(mapping, start=1, end=2)
    expected_result = {"b": 2}
    assert result == expected_result


def test_slice_mapping_size_overrides_start_end() -> None:
    mapping = {"a": 1, "b": 2, "c": 3}
    result = slice_mapping(mapping, size=2, start=0, end=2)
    expected_result = {"a": 1, "b": 2}
    assert result == expected_result


def test_slice_mapping_different_key_types() -> None:
    mapping = {1: "one", "two": 2, 3.0: "three"}
    result = slice_mapping(mapping, size=2)
    expected_result = {1: "one", "two": 2}
    assert result == expected_result


def test_slice_mapping_different_value_types() -> None:
    mapping = {"a": 1, "b": "two", "c": [3]}
    result = slice_mapping(mapping, start=1, end=2)
    expected_result = {"b": "two"}
    assert result == expected_result
