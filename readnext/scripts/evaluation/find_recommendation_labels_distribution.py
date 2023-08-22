import numpy as np

from readnext.config import DataPaths
from readnext.utils.io import read_df_from_parquet


def is_relevant_document(
    query_document_labels: list[str], candidate_document_labels: list[str]
) -> bool:
    return any(label in query_document_labels for label in candidate_document_labels)


def get_proportion_relevant_documents(
    query_document_labels: list[str], candidate_document_labels_list: list[list[str]]
) -> float:
    return np.mean(
        [
            is_relevant_document(query_document_labels, candidate_document_labels)
            for candidate_document_labels in candidate_document_labels_list
        ]
    )  # type: ignore


def main() -> None:
    documents_frame = read_df_from_parquet(DataPaths.merged.documents_frame)
    all_document_labels_list = documents_frame["arxiv_labels"].to_list()

    proportions = []

    for i, query_document_labels in enumerate(all_document_labels_list):
        # remove query document from all documents to get candidate documents
        candidate_document_labels_list = (
            all_document_labels_list[:i] + all_document_labels_list[i + 1 :]
        )

        proportions.append(
            get_proportion_relevant_documents(query_document_labels, candidate_document_labels_list)
        )

    mean_proportion = np.mean(proportions)
    print(f"Proportions of relevant documents: {mean_proportion:.3f}")
    # Mean Proportion of relevant documents: 0.280


if __name__ == "__main__":
    main()
