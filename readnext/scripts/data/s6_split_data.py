"""
Splits the documents frame into training, validation and test sets for the evaluation
stage.
"""

import polars as pl

from readnext.config import DataPaths, MagicNumbers
from readnext.data.data_split import DataSplitIndices
from readnext.utils.aliases import DocumentsFrame
from readnext.utils.io import write_df_to_parquet


def get_training_indices(documents_frame: DocumentsFrame) -> pl.Series:
    return documents_frame.sample(
        n=MagicNumbers.training_size,
        with_replacement=False,
        seed=MagicNumbers.data_split_seed,
    )["d3_document_id"]


def get_validation_indices(
    documents_frame: DocumentsFrame, training_indices: pl.Series
) -> pl.Series:
    return documents_frame.filter(~pl.col("d3_document_id").is_in(training_indices)).sample(
        n=MagicNumbers.validation_size,
        with_replacement=False,
        seed=MagicNumbers.data_split_seed,
    )["d3_document_id"]


def get_test_indices(
    documents_frame: DocumentsFrame, training_indices: pl.Series, validation_indices: pl.Series
) -> pl.Series:
    return documents_frame.filter(
        ~pl.col("d3_document_id").is_in(training_indices)
        & ~pl.col("d3_document_id").is_in(validation_indices)
    )["d3_document_id"]


def split_indices(documents_frame: DocumentsFrame) -> DataSplitIndices:
    training_indices = get_training_indices(documents_frame)
    validation_indices = get_validation_indices(documents_frame, training_indices)
    test_indices = get_test_indices(documents_frame, training_indices, validation_indices)

    return DataSplitIndices(
        train=training_indices.to_list(),
        validation=validation_indices.to_list(),
        test=test_indices.to_list(),
    )


def main() -> None:
    documents_frame: DocumentsFrame = pl.read_parquet(DataPaths.merged.documents_frame)

    data_split_indices = split_indices(documents_frame)

    training_set = documents_frame.filter(pl.col("d3_document_id").is_in(data_split_indices.train))
    validation_set = documents_frame.filter(
        pl.col("d3_document_id").is_in(data_split_indices.validation)
    )
    test_set = documents_frame.filter(pl.col("d3_document_id").is_in(data_split_indices.test))

    write_df_to_parquet(training_set, DataPaths.merged.documents_frame_train)
    write_df_to_parquet(validation_set, DataPaths.merged.documents_frame_validation)
    write_df_to_parquet(test_set, DataPaths.merged.documents_frame_test)


if __name__ == "__main__":
    main()
