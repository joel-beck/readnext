import pandas as pd

from readnext.config import DataPaths


def main() -> None:
    documents_authors_labels_references: pd.DataFrame = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_pkl
    )

    documents_authors_labels_citations_most_cited = documents_authors_labels_references.sort_values(
        by="citationcount_document", ascending=False
    ).iloc[: DataPaths.merged.most_cited_subset_size]

    documents_authors_labels_citations_most_cited.to_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )


if __name__ == "__main__":
    main()
