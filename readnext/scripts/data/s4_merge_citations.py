"""
Send requests to the semanticscholar API to obtain citation and references data.
Requires a private API key which is stored in a `SEMANTICSCHOLAR_API_KEY` environment
variable.Run multiple processes in parallel to speed up the process. Add the new data to
the existing dataframe.
"""

import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from readnext.config import DataPaths
from readnext.data import SemanticscholarRequest, SemanticScholarResponse
from readnext.utils import (
    get_semanticscholar_url_from_semanticscholar_id,
    read_df_from_parquet,
    write_df_to_parquet,
)


def get_paper_citations(semanticscholar_response: SemanticScholarResponse) -> list[str]:
    return [
        get_semanticscholar_url_from_semanticscholar_id(citation.get("paperId", None))
        for citation in semanticscholar_response.citations
        if citation.get("paperId", None) is not None
    ]


def get_paper_references(semanticscholar_response: SemanticScholarResponse) -> list[str]:
    return [
        get_semanticscholar_url_from_semanticscholar_id(reference.get("paperId", None))
        for reference in semanticscholar_response.references
        if reference.get("paperId", None) is not None
    ]


def main() -> None:
    tqdm.pandas()
    load_dotenv()

    output_columns = [
        "d3_document_id",
        "d3_author_id",
        "title",
        "author",
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "citations",
        "references",
        "abstract",
        "semanticscholar_id",
        "semanticscholar_url",
        "semanticscholar_tags",
        "arxiv_id",
        "arxiv_url",
        "arxiv_labels",
    ]

    # one GET request is necessary per document, each one takes ~1.5 seconds
    # process subsets of documents in parallel to speed up the process
    USE_SUBSET = True

    documents_authors_labels: pl.DataFrame = read_df_from_parquet(
        DataPaths.merged.documents_authors_labels
    )

    SUBSET_START = int(sys.argv[1]) if USE_SUBSET else 0
    SUBSET_SIZE = 5000
    SUBSET_END = SUBSET_START + SUBSET_SIZE if USE_SUBSET else len(documents_authors_labels)

    if USE_SUBSET:
        documents_authors_labels = documents_authors_labels.slice(
            offset=SUBSET_START, length=SUBSET_SIZE
        )

    semanticscholar_request = SemanticscholarRequest()

    documents_authors_labels_citations = documents_authors_labels.with_columns(
        semanticscholar_request=pl.col("semanticscholar_url").apply(
            semanticscholar_request.from_semanticscholar_url
        ),
        citations=pl.col("semanticscholar_request").apply(get_paper_citations),
        references=pl.col("semanticscholar_request").apply(get_paper_references),
    ).drop(columns=["semanticscholar_request"])

    # subtract one from subset end since upper bound of slice is exclusive
    write_df_to_parquet(
        documents_authors_labels_citations,
        Path(
            f"{DataPaths.merged.documents_authors_labels_citations_chunks_stem}_{SUBSET_START}_{SUBSET_END-1}.pkl"
        ),
    )


if __name__ == "__main__":
    # run script in parallel with different command line arguments with
    # `parallel --bar -a merge_citations_cli_arguments.txt python merge_citations.py`
    # where merge_citations_cli_arguments.txt contains the command line arguments
    # separated by newlines
    main()
