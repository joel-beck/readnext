from dataclasses import dataclass
from enum import Enum

import polars as pl
from typing_extensions import Self

from readnext.config import DataPaths
from readnext.utils.aliases import DocumentsFrame


@dataclass
class DataSplitIndices:
    """Collects d3 document indices for the training, validation and test sets."""

    train: list[int]
    validation: list[int]
    test: list[int]

    @classmethod
    def from_frames(cls) -> Self:
        documents_frame_train = pl.read_parquet(DataPaths.merged.documents_frame_train)
        documents_frame_validation = pl.read_parquet(DataPaths.merged.documents_frame_validation)
        documents_frame_test = pl.read_parquet(DataPaths.merged.documents_frame_test)

        return cls(
            train=documents_frame_train["d3_document_id"].to_list(),
            validation=documents_frame_validation["d3_document_id"].to_list(),
            test=documents_frame_test["d3_document_id"].to_list(),
        )


class DataSplit(Enum):
    FULL = "full"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def load_data_split(data_split: DataSplit) -> DocumentsFrame:
    match data_split:
        case DataSplit.FULL:
            return pl.read_parquet(DataPaths.merged.documents_frame)
        case DataSplit.TRAIN:
            return pl.read_parquet(DataPaths.merged.documents_frame_train)
        case DataSplit.VALIDATION:
            return pl.read_parquet(DataPaths.merged.documents_frame_validation)
        case DataSplit.TEST:
            return pl.read_parquet(DataPaths.merged.documents_frame_test)
        case _:
            raise ValueError(f"Unknown data split: {data_split}")
