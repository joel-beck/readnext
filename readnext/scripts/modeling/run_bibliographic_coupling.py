"""
Precompute and store bibliographic coupling scores for all documents.
"""

from time import perf_counter

import polars as pl
from polars.testing import assert_frame_equal

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import precompute_co_references
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    documents_frame = read_df_from_parquet(DataPaths.merged.documents_frame).head(100)

    start_1 = perf_counter()

    bibliographic_coupling_scores = precompute_co_references(documents_frame)

    stop_1 = perf_counter()

    # print the execution time formatted as MM:SS
    print(f"{(stop_1 - start_1) / 60:02.0f}:{(stop_1 - start_1) % 60:02.0f}")

    start_2 = perf_counter()

    documents_frame = documents_frame.lazy()
    query_frame = documents_frame.select(["d3_document_id", "references"]).rename(
        {"d3_document_id": "query_d3_document_id", "references": "query_references"}
    )
    candidate_frame = documents_frame.select(["d3_document_id", "references"]).rename(
        {"d3_document_id": "candidate_d3_document_id", "references": "candidate_references"}
    )

    result = (
        query_frame.join(candidate_frame, how="cross")
        .filter(pl.col("query_d3_document_id") != pl.col("candidate_d3_document_id"))
        .with_columns(concat=pl.col("query_references").list.concat(pl.col("candidate_references")))
        .with_columns(unique=pl.col("concat").list.unique())
        .with_columns(
            score=pl.col("concat").list.lengths() - pl.col("unique").list.lengths().cast(pl.Int64)
        )
        .select(["query_d3_document_id", "candidate_d3_document_id", "score"])
        .sort(by=["query_d3_document_id", "score"], descending=[False, True])
        .groupby("query_d3_document_id")
        .head(100)
        .collect()
    )

    stop_2 = perf_counter()

    # print the execution time formatted as MM:SS
    print(f"{(stop_2 - start_2) / 60:02.0f}:{(stop_2 - start_2) % 60:02.0f}")

    assert_frame_equal(bibliographic_coupling_scores, result)

    # write_df_to_parquet(
    #     bibliographic_coupling_scores,
    #     ResultsPaths.citation_models.bibliographic_coupling_scores_parquet,
    # )


if __name__ == "__main__":
    main()
