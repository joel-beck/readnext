from dataclasses import dataclass, field

import pandas as pd

from readnext.data import (
    SemanticscholarRequest,
    add_citation_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.evaluation.metrics import (
    CosineSimilarity,
    CountCommonCitations,
    CountCommonReferences,
)
from readnext.inference.attribute_getter.attribute_getter_base import AttributeGetter
from readnext.inference.attribute_getter.attribute_getter_unseen_base import (
    QueryCitationModelDataConstructor,
    QueryLanguageModelDataConstructor,
    SemanticScholarResponse,
)
from readnext.inference.attribute_getter.attribute_getter_unseen_language_models import (
    select_query_embedding_function,
)
from readnext.inference.document_identifier import DocumentIdentifier
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    DocumentScore,
    LanguageModelData,
)
from readnext.modeling.language_models import load_embeddings_from_choice
from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
    sort_document_scores,
)


@dataclass(kw_only=True)
class UnseenPaperAttributeGetter(AttributeGetter):
    semanticscholar_request: SemanticscholarRequest = field(init=False)
    response: SemanticScholarResponse = field(init=False)

    def __post_init__(self) -> None:
        self.semanticscholar_request = SemanticscholarRequest()
        self.response = self.send_semanticscholar_request()
        super().__post_init__()

    def send_semanticscholar_request(self) -> SemanticScholarResponse:
        if self.semanticscholar_id is not None:
            return self.semanticscholar_request.from_semanticscholar_id(self.semanticscholar_id)

        if self.semanticscholar_url is not None:
            return self.semanticscholar_request.from_semanticscholar_url(self.semanticscholar_url)

        if self.arxiv_id is not None:
            return self.semanticscholar_request.from_arxiv_id(self.arxiv_id)

        if self.arxiv_url is not None:
            return self.semanticscholar_request.from_arxiv_url(self.arxiv_url)

        raise ValueError("No identifier provided")

    def get_identifier_from_semanticscholar_id(self, semanticscholar_id: str) -> DocumentIdentifier:
        return DocumentIdentifier(
            d3_document_id=-1,
            semanticscholar_id=semanticscholar_id,
            semanticscholar_url=get_semanticscholar_url_from_semanticscholar_id(semanticscholar_id),
            arxiv_id=self.response.arxiv_id,
            arxiv_url=get_arxiv_url_from_arxiv_id(self.response.arxiv_id),
        )

    def get_identifier_from_semanticscholar_url(
        self, semanticscholar_url: str
    ) -> DocumentIdentifier:
        return DocumentIdentifier(
            d3_document_id=-1,
            semanticscholar_id=get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url),
            semanticscholar_url=semanticscholar_url,
            arxiv_id=self.response.arxiv_id,
            arxiv_url=get_arxiv_url_from_arxiv_id(self.response.arxiv_id),
        )

    def get_identifier_from_arxiv_id(self, arxiv_id: str) -> DocumentIdentifier:
        return DocumentIdentifier(
            d3_document_id=-1,
            semanticscholar_id=self.response.semanticscholar_id,
            semanticscholar_url=get_semanticscholar_url_from_semanticscholar_id(
                self.response.semanticscholar_id
            ),
            arxiv_id=arxiv_id,
            arxiv_url=get_arxiv_url_from_arxiv_id(arxiv_id),
        )

    def get_identifier_from_arxiv_url(self, arxiv_url: str) -> DocumentIdentifier:
        return DocumentIdentifier(
            d3_document_id=-1,
            semanticscholar_id=self.response.semanticscholar_id,
            semanticscholar_url=get_semanticscholar_url_from_semanticscholar_id(
                self.response.semanticscholar_id
            ),
            arxiv_id=get_arxiv_id_from_arxiv_url(arxiv_url),
            arxiv_url=arxiv_url,
        )

    def get_query_citation_urls(self) -> list[str]:
        return (
            [
                get_semanticscholar_url_from_semanticscholar_id(citation["paperId"])
                for citation in self.response.citations
            ]
            if self.response.citations is not None
            else []
        )

    def get_query_reference_urls(self) -> list[str]:
        return (
            [
                get_semanticscholar_url_from_semanticscholar_id(reference["paperId"])
                for reference in self.response.references
            ]
            if self.response.references is not None
            else []
        )

    def get_co_citation_analysis_scores(self) -> pd.DataFrame:
        common_citations_scores: list[DocumentScore] = []

        for d3_document_id, candidate_citation_urls in self.documents_data["citations"].items():
            document_info = DocumentInfo(d3_document_id=d3_document_id.item())  # type: ignore

            query_citation_urls = self.get_query_citation_urls()
            common_citation_urls = CountCommonCitations.count_common_values(
                query_citation_urls, candidate_citation_urls
            )

            document_score = DocumentScore(document_info=document_info, score=common_citation_urls)
            common_citations_scores.append(document_score)

        return pd.DataFrame(
            {"scores": [sort_document_scores(common_citations_scores)]}, index=[-1]
        ).rename_axis("document_id", axis="index")

    def get_bibliographic_coupling_scores(self) -> pd.DataFrame:
        common_references_scores: list[DocumentScore] = []

        for d3_document_id, candidate_reference_urls in self.documents_data["references"].items():
            document_info = DocumentInfo(d3_document_id=d3_document_id.item())  # type: ignore

            query_reference_urls = self.get_query_reference_urls()
            common_references = CountCommonReferences.count_common_values(
                query_reference_urls, candidate_reference_urls
            )

            document_score = DocumentScore(document_info=document_info, score=common_references)
            common_references_scores.append(document_score)

        return pd.DataFrame(
            {"scores": [sort_document_scores(common_references_scores)]}, index=[-1]
        ).rename_axis("document_id", axis="index")

    def get_citation_model_data(self) -> CitationModelData:
        citation_model_data_constructor = QueryCitationModelDataConstructor(
            response=self.response,
            d3_document_id=-1,
            documents_data=self.documents_data.pipe(add_citation_feature_rank_cols)
            .pipe(add_citation_feature_rank_cols)
            .pipe(set_missing_publication_dates_to_max_rank),
            co_citation_analysis_scores=self.get_co_citation_analysis_scores(),
            bibliographic_coupling_scores=self.get_bibliographic_coupling_scores(),
        )

        return CitationModelData.from_constructor(citation_model_data_constructor)

    def get_query_document_info(self) -> DocumentInfo:
        assert self.response.title is not None
        assert self.response.abstract is not None

        return DocumentInfo(
            d3_document_id=-1,
            title=self.response.title,
            abstract=self.response.abstract,
        )

    def get_cosine_similarities(self) -> pd.DataFrame:
        query_document_info = self.get_query_document_info()
        query_embedding_function = select_query_embedding_function(self.language_model_choice)

        query_embedding = query_embedding_function(query_document_info)
        candidate_embeddings: pd.DataFrame = load_embeddings_from_choice(self.language_model_choice)

        cosine_similarity_scores: list[DocumentScore] = []

        for candidate_document_id, candidate_embedding in zip(
            candidate_embeddings.index, candidate_embeddings["embedding"]
        ):
            candidate_document_info = DocumentInfo(d3_document_id=candidate_document_id)
            cosine_similarity = CosineSimilarity.score(query_embedding, candidate_embedding)

            document_score = DocumentScore(
                document_info=candidate_document_info, score=cosine_similarity
            )
            cosine_similarity_scores.append(document_score)

        return pd.DataFrame(
            {"scores": [sort_document_scores(cosine_similarity_scores)]}, index=[-1]
        ).rename_axis("document_id", axis="index")

    def get_language_model_data(self) -> LanguageModelData:
        language_model_data_constructor = QueryLanguageModelDataConstructor(
            response=self.response,
            d3_document_id=-1,
            documents_data=self.documents_data,
            cosine_similarities=self.get_cosine_similarities(),
        )

        return LanguageModelData.from_constructor(language_model_data_constructor)
