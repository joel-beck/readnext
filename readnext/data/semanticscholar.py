import os
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from typing_extensions import TypedDict

from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_id_from_semanticscholar_url,
)


class SemanticScholarCitation(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None


class SemanticScholarReference(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None


class ExternalIds(TypedDict):
    ArXiv: str | None
    DBLP: str | None
    PubMedCentral: str | None


class SemanticScholarJson(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None
    abstract: str | None
    citations: list[SemanticScholarCitation]
    references: list[SemanticScholarReference]
    externalIds: ExternalIds  # noqa: N815


@dataclass(kw_only=True)
class SemanticScholarResponse:
    semanticscholar_id: str
    arxiv_id: str
    title: str
    abstract: str
    citations: list[SemanticScholarCitation]
    references: list[SemanticScholarReference]


@dataclass
class SemanticscholarRequest:
    semanticscholar_api_key: str = field(init=False)
    request_headers: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        load_dotenv()
        self.semanticscholar_api_key = os.getenv("SEMANTICSCHOLAR_API_KEY", "")
        self.request_headers = {"x-api-key": self.semanticscholar_api_key}

    def get_request_url_from_semanticscholar_id(self, semanticscholar_id: str) -> str:
        return f"https://api.semanticscholar.org/graph/v1/paper/{semanticscholar_id}?fields=abstract,citations,references,title"

    def get_request_url_from_arxiv_id(self, arxiv_id: str) -> str:
        return f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=abstract,citations,references,title"

    def get_response_from_request_url(self, request_url: str) -> SemanticScholarResponse:
        response: SemanticScholarJson = requests.get(
            request_url, headers=self.request_headers
        ).json()

        arxiv_id = ""
        if (external_ids := response.get("externalIds", None)) is not None:
            arxiv_id = external_ids["ArXiv"] if external_ids["ArXiv"] is not None else ""

        return SemanticScholarResponse(
            semanticscholar_id=response["paperId"] if response["paperId"] is not None else "",
            arxiv_id=arxiv_id,
            title=response["title"] if response["title"] is not None else "",
            abstract=response["abstract"] if response["abstract"] is not None else "",
            citations=response["citations"],
            references=response["references"],
        )

    def from_semanticscholar_id(self, semanticscholar_id: str) -> SemanticScholarResponse:
        request_url = self.get_request_url_from_semanticscholar_id(semanticscholar_id)
        return self.get_response_from_request_url(request_url)

    def from_semanticscholar_url(self, semanticscholar_url: str) -> SemanticScholarResponse:
        semanticscholar_id = get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url)
        request_url = self.get_request_url_from_semanticscholar_id(semanticscholar_id)
        return self.get_response_from_request_url(request_url)

    def from_arxiv_id(self, arxiv_id: str) -> SemanticScholarResponse:
        request_url = self.get_request_url_from_arxiv_id(arxiv_id)
        return self.get_response_from_request_url(request_url)

    def from_arxiv_url(self, arxiv_url: str) -> SemanticScholarResponse:
        arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)
        request_url = self.get_request_url_from_arxiv_id(arxiv_id)
        return self.get_response_from_request_url(request_url)
