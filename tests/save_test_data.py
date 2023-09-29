"""
Read the first 10 rows of all important data files and store them in the project test
folder to use for testing.
"""

from dataclasses import fields, is_dataclass
from pathlib import Path

from readnext.config import DataPaths, MagicNumbers, ResultsPaths
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


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
    testing_data_dirpath = Path(__file__).parent / "testing_data"

    documents_frame_path = DataPaths.merged.documents_frame
    results_paths = get_all_paths_from_dataclass(ResultsPaths)

    all_paths = [documents_frame_path, *results_paths]

    for path in all_paths:
        destination_path = testing_data_dirpath / f"test_{path.name}"

        if destination_path.suffix != ".parquet":
            continue

        df = read_df_from_parquet(path)
        write_df_to_parquet(df.head(MagicNumbers.unit_testing_size), destination_path)


if __name__ == "__main__":
    main()
