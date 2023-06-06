"""
Precompute and store co-citation scores for all documents in a dataframe.
"""

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import precompute_co_citations
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    documents_frame = read_df_from_parquet(DataPaths.merged.documents_frame)
    # NOTE: Remove to train on full data
    documents_frame = documents_frame.head(1000)

    co_citation_analysis_scores = precompute_co_citations(documents_frame)

    write_df_to_parquet(
        co_citation_analysis_scores,
        ResultsPaths.citation_models.co_citation_analysis_scores_parquet,
    )


if __name__ == "__main__":
    main()
