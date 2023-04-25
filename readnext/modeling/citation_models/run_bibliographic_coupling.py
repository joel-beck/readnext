import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation import precompute_co_references


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )

    bibliographic_coupling_most_cited = precompute_co_references(
        documents_authors_labels_citations_most_cited
    )

    bibliographic_coupling_most_cited.to_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_most_cited_pkl
    )


if __name__ == "__main__":
    main()
