"""
Precompute and store co-citation scores for all documents in a dataframe.
"""

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import precompute_co_citations
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    documents_authors_labels_citations_most_cited = read_df_from_parquet(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    documents_authors_labels_citations_most_cited = (
        documents_authors_labels_citations_most_cited.head(1000)
    )

    co_citation_analysis_scores_most_cited = precompute_co_citations(
        documents_authors_labels_citations_most_cited
    )

    write_df_to_parquet(
        co_citation_analysis_scores_most_cited,
        ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl,
    )


if __name__ == "__main__":
    main()
