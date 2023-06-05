"""
Precompute and store bibliographic coupling scores for all documents in a dataframe.
"""

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import precompute_co_references
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    documents_data = read_df_from_parquet(DataPaths.merged.documents_data)
    # NOTE: Remove to train on full data
    documents_data = documents_data.head(1000)

    bibliographic_coupling_scores = precompute_co_references(documents_data)

    write_df_to_parquet(
        bibliographic_coupling_scores,
        ResultsPaths.citation_models.bibliographic_coupling_scores_parquet,
    )


if __name__ == "__main__":
    main()
