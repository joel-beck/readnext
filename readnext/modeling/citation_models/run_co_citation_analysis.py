import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.pairwise_scores import precompute_co_citations


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )

    # option 1: pre-compute all pairwise counts, computation at training time
    co_citation_analysis_most_cited = precompute_co_citations(
        documents_authors_labels_citations_most_cited
    )

    co_citation_analysis_most_cited.to_pickle(
        ResultsPaths.citation_models.co_citation_analysis_most_cited_pkl
    )


if __name__ == "__main__":
    main()
