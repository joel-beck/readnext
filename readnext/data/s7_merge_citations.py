"""
Send requests to the semanticscholar API to obtain citation and references data.
Requires a private API key which is stored in a `SEMANTICSCHOLAR_API_KEY` environment
variable.Run multiple processes in parallel to speed up the process. Add the new data to
the existing dataframe.
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from typing_extensions import TypedDict

from readnext.config import DataPaths


class SemanticScholarCitation(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None


class SemanticScholarReference(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None


class SemanticScholarJson(TypedDict):
    paperId: str | None  # noqa: N815
    citations: list[SemanticScholarCitation]
    references: list[SemanticScholarReference]


@dataclass
class SemanticScholarResponse:
    paper_id: str | None
    citations: list[SemanticScholarCitation] | None
    references: list[SemanticScholarReference] | None


def get_id_from_url(semanticscholar_url: str) -> str:
    """Paper ID is the last part of the URL after the final forward slash"""
    return semanticscholar_url.rsplit("/", 1)[-1]


def get_url_from_id(semanticscholar_id: str | None) -> str:
    if semanticscholar_id is None:
        return ""

    return f"https://semanticscholar.org/paper/{semanticscholar_id}"


def send_semanticscholar_request(
    semanticscholar_id: str, headers: dict[str, str]
) -> SemanticScholarResponse:
    response: SemanticScholarJson = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{semanticscholar_id}?fields=citations,references",
        headers=headers,
    ).json()
    return SemanticScholarResponse(
        paper_id=response.get("paperId", None),
        citations=response.get("citations", None),
        references=response.get("references", None),
    )


def get_paper_citations(semanticscholar_response: SemanticScholarResponse) -> list[str]:
    if semanticscholar_response.citations is None:
        return []

    return [
        get_url_from_id(citation.get("paperId", None))
        for citation in semanticscholar_response.citations
        if citation.get("paperId", None) is not None
    ]


def get_paper_references(semanticscholar_response: SemanticScholarResponse) -> list[str]:
    if semanticscholar_response.references is None:
        return []

    return [
        get_url_from_id(reference.get("paperId", None))
        for reference in semanticscholar_response.references
        if reference.get("paperId", None) is not None
    ]


def main() -> None:
    # one GET request is necessary per document, each one takes ~1.5 seconds
    # process subsets of documents in parallel to speed up the process
    USE_SUBSET = True

    SUBSET_START = int(sys.argv[1])
    SUBSET_SIZE = 5000
    SUBSET_END = SUBSET_START + SUBSET_SIZE

    tqdm.pandas()
    load_dotenv()

    SEMANTICSCHOLAR_API_KEY = os.getenv("SEMANTICSCHOLAR_API_KEY", "")
    request_headers = {"x-api-key": SEMANTICSCHOLAR_API_KEY}

    documents_authors_labels: pd.DataFrame = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_pkl
    )
    if USE_SUBSET:
        documents_authors_labels = documents_authors_labels.iloc[SUBSET_START:SUBSET_END]

    documents_authors_labels_citations = documents_authors_labels.assign(
        semanticscholar_request=lambda df: df["semanticscholar_url"].progress_apply(
            lambda url: send_semanticscholar_request(get_id_from_url(url), request_headers)
        ),
        citations=lambda df: df["semanticscholar_request"].apply(get_paper_citations),
        references=lambda df: df["semanticscholar_request"].apply(get_paper_references),
    ).drop(columns=["semanticscholar_request"])

    # subtract one from subset end since upper bound of slice is exclusive
    documents_authors_labels_citations.to_pickle(
        f"{DataPaths.merged.documents_authors_labels_citations_chunks_stem}_{SUBSET_START}_{SUBSET_END-1}.pkl"
    )


if __name__ == "__main__":
    # run script in parallel with different command line arguments with
    # `parallel --bar -a merge_citations_cli_arguments.txt python merge_citations.py`
    # where merge_citations_cli_arguments.txt contains the command line arguments
    # separated by newlines
    main()
