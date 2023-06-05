from dataclasses import dataclass, field

import polars as pl

from readnext.data import SemanticscholarRequest, SemanticScholarResponse
from readnext.evaluation.metrics import (
    CosineSimilarity,
    CountCommonCitations,
    CountCommonReferences,
)
from readnext.inference.constructor_plugin import InferenceDataConstructorPlugin
from readnext.inference.constructor_plugin_unseen_language_models import (
    select_query_embedding_function,
)
from readnext.inference.document_identifier import DocumentIdentifier
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelData,
    LanguageModelDataConstructor,
    UnseenModelDataConstructorPlugin,
)
from readnext.modeling.language_models import load_embeddings_from_choice
from readnext.utils import (
    CandidateScoresFrame,
    get_arxiv_id_from_arxiv_url,
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
)


@dataclass(kw_only=True)
class UnseenInferenceDataConstructorPlugin(InferenceDataConstructorPlugin):
    """
    `InferenceDataConstructor` methods for unseen query documents only.
    """

    semanticscholar_request: SemanticscholarRequest = field(init=False)
    response: SemanticScholarResponse = field(init=False)
    model_data_constructor_plugin: UnseenModelDataConstructorPlugin = field(init=False)

    def __post_init__(self) -> None:
        self.semanticscholar_request = SemanticscholarRequest()
        self.response = self.send_semanticscholar_request()
        self.model_data_constructor_plugin = UnseenModelDataConstructorPlugin(
            response=self.response,
        )
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

    @staticmethod
    def select_scoring_input_columns(df: pl.DataFrame, colname: str) -> pl.DataFrame:
        return df.select(["d3_document_id", colname]).rename(
            {
                "d3_document_id": "candidate_d3_document_id",
            }
        )

    @staticmethod
    def select_scoring_output_columns(df: pl.DataFrame) -> CandidateScoresFrame:
        return df.select("candidate_d3_document_id", "score").sort("score", descending=True)

    def get_co_citation_analysis_scores(self) -> CandidateScoresFrame:
        return (
            self.documents_data.pipe(self.select_scoring_input_columns, "citations")
            .with_columns(query_citations=self.get_query_citation_urls())
            .with_columns(
                score=pl.struct(["citations", "query_citations"]).apply(
                    lambda df: CountCommonCitations.count_common_values(
                        df["citations"], df["query_citations"]  # type: ignore
                    ),
                ),
            )
            .pipe(self.select_scoring_output_columns)
        )

    def get_bibliographic_coupling_scores(self) -> CandidateScoresFrame:
        return (
            self.documents_data.pipe(self.select_scoring_input_columns, "references")
            .with_columns(query_references=self.get_query_reference_urls())
            .with_columns(
                score=pl.struct(["references", "query_references"]).apply(
                    lambda df: CountCommonReferences.count_common_values(
                        df["references"], df["query_references"]  # type: ignore
                    ),
                ),
            )
            .pipe(self.select_scoring_output_columns)
        )

    def get_citation_model_data(self) -> CitationModelData:
        citation_model_data_constructor = CitationModelDataConstructor(
            d3_document_id=-1,
            documents_data=self.documents_data,
            constructor_plugin=self.model_data_constructor_plugin,
            co_citation_analysis_scores_frame=self.get_co_citation_analysis_scores(),
            bibliographic_coupling_scores_frame=self.get_bibliographic_coupling_scores(),
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

    def get_query_documents_data(self) -> pl.DataFrame:
        query_document_info = self.get_query_document_info()
        return pl.DataFrame(
            {
                "d3_document_id": query_document_info.d3_document_id,
                "title": query_document_info.title,
                "abstract": query_document_info.abstract,
            }
        )

    def get_cosine_similarities(self) -> CandidateScoresFrame:
        query_document_data = self.get_query_documents_data()
        query_embedding_function = select_query_embedding_function(self.language_model_choice)

        query_embedding = query_embedding_function(query_document_data)
        candidate_embeddings: pl.DataFrame = load_embeddings_from_choice(self.language_model_choice)

        return (
            candidate_embeddings.pipe(self.select_scoring_input_columns, "embedding")
            .with_columns(query_embedding=query_embedding)
            .with_columns(
                score=pl.struct(["embedding", "query_embedding"]).apply(
                    lambda df: CosineSimilarity.score(df["embedding"], df["query_embedding"]),  # type: ignore # noqa: E501
                ),
            )
            .pipe(self.select_scoring_output_columns)
        )

    def get_language_model_data(self) -> LanguageModelData:
        language_model_data_constructor = LanguageModelDataConstructor(
            d3_document_id=-1,
            documents_data=self.documents_data,
            constructor_plugin=self.model_data_constructor_plugin,
            cosine_similarity_scores_frame=self.get_cosine_similarities(),
        )

        return LanguageModelData.from_constructor(language_model_data_constructor)
