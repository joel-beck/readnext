import os
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired

import requests
from dotenv import load_dotenv
from joblib import Parallel, delayed

from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_id_from_semanticscholar_url,
    setup_progress_bar,
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
    paperId: NotRequired[str | None]  # noqa: N815
    title: NotRequired[str | None]
    abstract: NotRequired[str | None]
    citations: NotRequired[list[SemanticScholarCitation]]
    references: NotRequired[list[SemanticScholarReference]]
    externalIds: NotRequired[ExternalIds | None]  # noqa: N815


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

    def send_semanticscholar_request(self, request_url: str) -> SemanticScholarJson:
        return requests.get(request_url, headers=self.request_headers).json()

    def extract_key_from_json_response(
        self,
        json_response: SemanticScholarJson,
        key: Literal["paperId", "title", "abstract", "citations", "references", "externalIds"],
        default_value: str | list[str] = "",
    ) -> Any:
        return value if (value := json_response.get(key)) is not None else default_value

    def extract_semanticscholar_id_from_json_response(
        self, json_response: SemanticScholarJson
    ) -> str:
        return self.extract_key_from_json_response(json_response, "paperId")

    def extract_arxiv_id_from_json_response(self, json_response: SemanticScholarJson) -> str:
        if (external_ids := json_response.get("externalIds")) is None:
            return ""

        return arxiv_id if (arxiv_id := external_ids.get("ArXiv")) is not None else ""

    def extract_title_from_json_response(self, json_response: SemanticScholarJson) -> str:
        return self.extract_key_from_json_response(json_response, "title")

    def extract_abstract_from_json_response(self, json_response: SemanticScholarJson) -> str:
        return self.extract_key_from_json_response(json_response, "abstract")

    def extract_citations_from_json_response(
        self, json_response: SemanticScholarJson
    ) -> list[SemanticScholarCitation]:
        return self.extract_key_from_json_response(json_response, "citations", default_value=[])

    def extract_references_from_json_response(
        self, json_response: SemanticScholarJson
    ) -> list[SemanticScholarReference]:
        return self.extract_key_from_json_response(json_response, "references", default_value=[])

    def get_response_from_request(
        self, json_response: SemanticScholarJson
    ) -> SemanticScholarResponse:
        return SemanticScholarResponse(
            semanticscholar_id=self.extract_semanticscholar_id_from_json_response(json_response),
            arxiv_id=self.extract_arxiv_id_from_json_response(json_response),
            title=self.extract_title_from_json_response(json_response),
            abstract=self.extract_abstract_from_json_response(json_response),
            citations=self.extract_citations_from_json_response(json_response),
            references=self.extract_references_from_json_response(json_response),
        )

    def get_response_from_request_url(self, request_url: str) -> SemanticScholarResponse:
        json_response = self.send_semanticscholar_request(request_url)
        return self.get_response_from_request(json_response)

    def from_semanticscholar_id(self, semanticscholar_id: str) -> SemanticScholarResponse:
        request_url = self.get_request_url_from_semanticscholar_id(semanticscholar_id)
        return self.get_response_from_request_url(request_url)

    def from_semanticscholar_url(self, semanticscholar_url: str) -> SemanticScholarResponse:
        semanticscholar_id = get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url)
        request_url = self.get_request_url_from_semanticscholar_id(semanticscholar_id)
        return self.get_response_from_request_url(request_url)

    def from_semanticscholar_urls(
        self, semanticscholar_urls: list[str], n_jobs: int = -1
    ) -> list[SemanticScholarResponse]:
        """
        Send GET requests for multiple semanticscholar_urls in parallel.
        """
        with setup_progress_bar() as progress_bar:
            return Parallel(n_jobs=n_jobs)(
                delayed(self.from_semanticscholar_url)(url)
                for url in progress_bar.track(
                    semanticscholar_urls,
                    total=len(semanticscholar_urls),
                    description="Sending Requests...",
                )
            )  # type: ignore

    def from_arxiv_id(self, arxiv_id: str) -> SemanticScholarResponse:
        request_url = self.get_request_url_from_arxiv_id(arxiv_id)
        return self.get_response_from_request_url(request_url)

    def from_arxiv_url(self, arxiv_url: str) -> SemanticScholarResponse:
        arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)
        request_url = self.get_request_url_from_arxiv_id(arxiv_id)
        return self.get_response_from_request_url(request_url)
