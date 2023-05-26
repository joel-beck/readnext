from dataclasses import dataclass

from readnext.data import SemanticScholarResponse
from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelDataConstructor,
)


def overwrite_collect_query_document(response: SemanticScholarResponse) -> DocumentInfo:
    title = response.title if response.title is not None else ""
    abstract = response.abstract if response.abstract is not None else ""

    return DocumentInfo(d3_document_id=-1, title=title, abstract=abstract)


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
