"""
Precompute and store bibliographic coupling scores for all documents.
"""


import polars as pl

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import precompute_co_references_polars
from readnext.utils import write_df_to_parquet


def main() -> None:
    documents_frame = pl.scan_parquet(DataPaths.merged.documents_frame)

    bibliographic_coupling_scores = precompute_co_references_polars(documents_frame)

    write_df_to_parquet(
        bibliographic_coupling_scores,
        ResultsPaths.citation_models.bibliographic_coupling_scores_parquet,
    )


if __name__ == "__main__":
    main()
