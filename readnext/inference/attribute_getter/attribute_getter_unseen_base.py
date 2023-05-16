from dataclasses import dataclass

import requests
from typing_extensions import TypedDict

from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelDataConstructor,
)


class SemanticScholarCitation(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None


class SemanticScholarReference(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None


class SemanticScholarJson(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None
    abstract: str | None
    citations: list[SemanticScholarCitation]
    references: list[SemanticScholarReference]


@dataclass
class SemanticScholarResponse:
    paper_id: str | None
    title: str | None
    abstract: str | None
    citations: list[SemanticScholarCitation] | None
    references: list[SemanticScholarReference] | None


@dataclass(kw_only=True)
class QueryCitationModelDataConstructor(CitationModelDataConstructor):
    response: SemanticScholarResponse

    def collect_query_document(self) -> DocumentInfo:
        return overwrite_collect_query_document(self.response)


@dataclass(kw_only=True)
class QueryLanguageModelDataConstructor(LanguageModelDataConstructor):
    response: SemanticScholarResponse

    def collect_query_document(self) -> DocumentInfo:
        return overwrite_collect_query_document(self.response)


def overwrite_collect_query_document(response: SemanticScholarResponse) -> DocumentInfo:
    title = response.title if response.title is not None else ""
    abstract = response.abstract if response.abstract is not None else ""

    return DocumentInfo(document_id=-1, title=title, abstract=abstract)


def get_request_url_from_semanticscholar_id(semanticscholar_id: str) -> str:
    return f"https://api.semanticscholar.org/graph/v1/paper/{semanticscholar_id}?fields=abstract,citations,references,title"


def get_request_url_from_arxiv_id(arxiv_id: str) -> str:
    return f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=abstract,citations,references,title"


def get_request_url_from_input(
    semanticsholar_id: str | None = None, arxiv_id: str | None = None
) -> str:
    if semanticsholar_id is not None:
        return get_request_url_from_semanticscholar_id(semanticsholar_id)

    if arxiv_id is not None:
        return get_request_url_from_arxiv_id(arxiv_id)

    raise ValueError("Either semanticsholar_id or arxiv_id must be provided")


def send_semanticscholar_request(
    *,
    semanticscholar_id: str | None = None,
    arxiv_id: str | None = None,
    request_headers: dict[str, str],
) -> SemanticScholarResponse:
    request_url = get_request_url_from_input(
        semanticsholar_id=semanticscholar_id, arxiv_id=arxiv_id
    )
    response: SemanticScholarJson = requests.get(request_url, headers=request_headers).json()

    return SemanticScholarResponse(
        paper_id=response.get("paperId", None),
        title=response.get("title", None),
        abstract=response.get("abstract", None),
        citations=response.get("citations", None),
        references=response.get("references", None),
    )
