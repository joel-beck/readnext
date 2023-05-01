"""
Precompute and store co-citation scores for all documents in a dataframe.
"""

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import precompute_co_citations


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    documents_authors_labels_citations_most_cited = (
        documents_authors_labels_citations_most_cited.head(100)
    )

    co_citation_analysis_scores_most_cited = precompute_co_citations(
        documents_authors_labels_citations_most_cited
    )

    co_citation_analysis_scores_most_cited.to_pickle(
        ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
    )


if __name__ == "__main__":
    main()
