import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation import precompute_co_references
from readnext.utils import save_df_to_pickle


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    documents_authors_labels_citations_most_cited = (
        documents_authors_labels_citations_most_cited.head(1000)
    )

    bibliographic_coupling_scores_most_cited = precompute_co_references(
        documents_authors_labels_citations_most_cited
    )

    save_df_to_pickle(
        bibliographic_coupling_scores_most_cited,
        ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl,
    )


if __name__ == "__main__":
    main()
