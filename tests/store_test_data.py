"""
Read the first 10 rows of all important data files and store them in the project test
folder to use for testing.
"""

from dataclasses import fields, is_dataclass
from pathlib import Path

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.utils import (
    load_df_from_pickle,
    slice_mapping,
    write_df_to_pickle,
    write_object_to_pickle,
)


def get_all_paths_from_dataclass(dataclass: object, paths: list[Path] | None = None) -> list[Path]:
    if not is_dataclass(dataclass):
        raise TypeError(f"Expected dataclass, got {type(dataclass)}")

    if paths is None:
        paths = []

    for field in fields(dataclass):
        # if default value is a Path, add it to the list of paths
        if isinstance(path := field.default, Path):
            paths.append(path)

        # if default value is a Dataclass itself, call function recursively on that dataclass
        if is_dataclass(nested_dataclass := field.default):
            get_all_paths_from_dataclass(nested_dataclass, paths)

    return paths


def main() -> None:
    # NOTE: This number must be the same as the value of the `test_data_size()` fixture in
    # `conftest.py`
    TEST_DATA_SIZE = 100
    test_data_dirpath = Path(__file__).parent / "data"

    documents_data_most_cited_path = (
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )
    results_paths = get_all_paths_from_dataclass(ResultsPaths)

    all_paths = [documents_data_most_cited_path, *results_paths]

    for path in all_paths:
        destination_path = test_data_dirpath / f"test_{path.name}"

        # consider only pickle files
        if destination_path.suffix != ".pkl":
            continue

        obj = load_df_from_pickle(path)

        if isinstance(obj, pd.DataFrame):
            write_df_to_pickle(obj.head(TEST_DATA_SIZE), destination_path)
            continue

        if isinstance(obj, dict):
            write_object_to_pickle(slice_mapping(obj, size=TEST_DATA_SIZE), destination_path)


if __name__ == "__main__":
    main()
