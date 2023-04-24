import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.pairwise_scores import precompute_co_references


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )

    # takes roughly 2 minutes
    bibliographic_coupling_most_cited = precompute_co_references(
        documents_authors_labels_citations_most_cited
    )

    bibliographic_coupling_most_cited.to_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_most_cited_pkl
    )

    # 8 MB of memory usage
    # references_df.memory_usage(deep=True).sum() / 1e6


if __name__ == "__main__":
    main()
