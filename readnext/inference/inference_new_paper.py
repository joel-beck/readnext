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
import spacy
from readnext.config import DataPaths, ModelVersions
from readnext.utils import (
    get_semanticscholar_id_from_semanticscholar_url,
    get_arxiv_id_from_arxiv_url,
    load_df_from_pickle,
    get_semanticscholar_url_from_semanticscholar_id,
)
from readnext.evaluation.metrics import CountCommonCitations, CountCommonReferences
from readnext.modeling.language_models import TFIDFEmbedder, SpacyTokenizer, tfidf
from readnext.modeling import DocumentInfo, DocumentsInfo


class SemanticScholarCitation(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None


class SemanticScholarReference(TypedDict):
    paperId: str | None  # noqa: N815
    title: str | None


class SemanticScholarJson(TypedDict):
    paperId: str | None  # noqa: N815
    abstract: str | None
    citations: list[SemanticScholarCitation]
    references: list[SemanticScholarReference]


@dataclass
class SemanticScholarResponse:
    paper_id: str | None
    abstract: str | None
    citations: list[SemanticScholarCitation] | None
    references: list[SemanticScholarReference] | None


def get_request_url_from_semanticscholar_id(semanticscholar_id: str) -> str:
    return f"https://api.semanticscholar.org/graph/v1/paper/{semanticscholar_id}?fields=citations,references,abstract"


def get_request_url_from_arxiv_id(arxiv_id: str) -> str:
    return f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}?fields=citations,references,abstract"


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
        abstract=response.get("abstract", None),
        citations=response.get("citations", None),
        references=response.get("references", None),
    )


# BOOKMARK: Inputs
semanticscholar_url = (
    "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
)
arxiv_url = "https://arxiv.org/abs/1706.03762"

load_dotenv()

documents_authors_labels_citations_most_cited: pd.DataFrame = load_df_from_pickle(
    DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
).set_index("document_id")

SEMANTICSCHOLAR_API_KEY = os.getenv("SEMANTICSCHOLAR_API_KEY", "")
request_headers = {"x-api-key": SEMANTICSCHOLAR_API_KEY}

# with semanticscholar_url
response = send_semanticscholar_request(
    semanticscholar_id=get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url),
    request_headers=request_headers,
)

# with arxiv_url
response = send_semanticscholar_request(
    arxiv_id=get_arxiv_id_from_arxiv_url(arxiv_url), request_headers=request_headers
)

assert response.paper_id is not None
assert response.abstract is not None

query_document_info = DocumentInfo(
    document_id=int(response.paper_id),
    abstract=response.abstract,
)
query_documents_info = DocumentsInfo(documents_info=[query_document_info])

query_citation_urls = (
    [
        get_semanticscholar_url_from_semanticscholar_id(citation["paperId"])
        for citation in response.citations
    ]
    if response.citations is not None
    else []
)

reference_urls = (
    [
        get_semanticscholar_url_from_semanticscholar_id(reference["paperId"])
        for reference in response.references
    ]
    if response.references is not None
    else []
)


num_common_citations: list[int] = [
    CountCommonCitations.count_common_values(query_citation_urls, candidate_citation_urls)
    for candidate_citation_urls in documents_authors_labels_citations_most_cited["citations"]
]

num_common_references: list[int] = [
    CountCommonReferences.count_common_values(reference_urls, candidate_reference_urls)
    for candidate_reference_urls in documents_authors_labels_citations_most_cited["references"]
]

spacy_model = spacy.load(ModelVersions.spacy)
spacy_tokenizer = SpacyTokenizer(documents_info=query_documents_info, spacy_model=spacy_model)

query_abstract_tokens_mapping = spacy_tokenizer.tokenize()

tfidf_embedder = TFIDFEmbedder(keyword_algorithm=tfidf)
tfidf_embeddings_mapping = tfidf_embedder.compute_embeddings_mapping(query_abstract_tokens_mapping)
