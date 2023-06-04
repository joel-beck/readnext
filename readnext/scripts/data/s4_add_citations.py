"""
Send requests to the semanticscholar API to obtain citation and reference urls for all
documents in the dataset and add them as features to the dataframe.

Requires a private API key which is stored in the `SEMANTICSCHOLAR_API_KEY` environment
variable.
"""

import polars as pl

from readnext.config import DataPaths
from readnext.data import SemanticscholarRequest, SemanticScholarResponse
from readnext.utils import (
    get_semanticscholar_url_from_semanticscholar_id,
    read_df_from_parquet,
    write_df_to_parquet,
)


def get_citation_urls_from_response(semanticscholar_response: SemanticScholarResponse) -> list[str]:
    return [
        get_semanticscholar_url_from_semanticscholar_id(paper_id)
        for citation in semanticscholar_response.citations
        if (paper_id := citation.get("paperId", None)) is not None
    ]


def get_reference_urls_from_response(
    semanticscholar_response: SemanticScholarResponse,
) -> list[str]:
    return [
        get_semanticscholar_url_from_semanticscholar_id(paper_id)
        for reference in semanticscholar_response.references
        if (paper_id := reference.get("paperId", None)) is not None
    ]


def send_semanticscholar_requests(
    df: pl.DataFrame, semanticscholar_request: SemanticscholarRequest
) -> list[SemanticScholarResponse]:
    """
    Send a semanticscholar request for each document in the dataframe.
    """
    semanticscholar_urls = df["semanticscholar_url"].to_list()
    return semanticscholar_request.from_semanticscholar_urls(semanticscholar_urls)


def add_semanticscholar_features(
    df: pl.DataFrame, semanticscholar_request: SemanticscholarRequest
) -> pl.DataFrame:
    """
    Add citation and reference urls to the dataframe.
    """
    responses = send_semanticscholar_requests(df, semanticscholar_request)
    citation_urls = [get_citation_urls_from_response(response) for response in responses]
    reference_urls = [get_reference_urls_from_response(response) for response in responses]

    return df.with_columns(
        citations=pl.Series(citation_urls),
        references=pl.Series(reference_urls),
    )


def main() -> None:
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

    # need eager mode for sending requests
    documents_authors_labels = read_df_from_parquet(DataPaths.merged.documents_authors_labels)

    semanticscholar_request = SemanticscholarRequest()

    documents_authors_labels_citations = documents_authors_labels.pipe(
        add_semanticscholar_features, semanticscholar_request
    ).select(output_columns)

    write_df_to_parquet(
        documents_authors_labels_citations, DataPaths.merged.documents_authors_labels_citations
    )


if __name__ == "__main__":
    main()
