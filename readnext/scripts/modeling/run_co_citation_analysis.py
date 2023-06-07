"""
Precompute and store co-citation analysis scores for all documents.
"""

import polars as pl

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import precompute_co_citations_polars
from readnext.utils import write_df_to_parquet


def main() -> None:
    documents_frame = pl.scan_parquet(DataPaths.merged.documents_frame)

    co_citation_analysis_scores = precompute_co_citations_polars(documents_frame)

    write_df_to_parquet(
        co_citation_analysis_scores,
        ResultsPaths.citation_models.co_citation_analysis_scores_parquet,
    )


if __name__ == "__main__":
    main()
