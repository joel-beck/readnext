from dataclasses import dataclass, field

from readnext.config import ResultsPaths
from readnext.inference.constructor_plugin import (
    DocumentIdentifier,
    InferenceDataConstructorPlugin,
)
from readnext.inference.input_converter import InferenceDataInputConverter
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
    SeenModelDataConstructorPlugin,
)
from readnext.modeling.language_models import (
    load_cosine_similarities_from_choice,
)
from readnext.utils import ScoresFrame, read_df_from_parquet


@dataclass(kw_only=True)
class SeenInferenceDataConstructorPlugin(InferenceDataConstructorPlugin):
    """
    `InferenceDataConstructor` methods for seen query documents only.
    """

    input_converter: InferenceDataInputConverter = field(init=False)
    model_data_constructor_plugin: SeenModelDataConstructorPlugin = field(init=False)

    def __post_init__(self) -> None:
        # must be called *before* `super().__post_init__()` since
        # `super().__post_init__()` already used the input converter
        self.input_converter = InferenceDataInputConverter(documents_frame=self.documents_frame)

        super().__post_init__()

        # must be called *after* `super().__post_init__()` since it requires the
        # `identifier` and `documents_frame` attributes from the parent class
        self.model_data_constructor_plugin = SeenModelDataConstructorPlugin(
            d3_document_id=self.identifier.d3_document_id,
            documents_frame=self.documents_frame,
        )

    def get_identifier_from_semanticscholar_id(self, semanticscholar_id: str) -> DocumentIdentifier:
        d3_document_id = self.input_converter.get_d3_document_id_from_semanticscholar_id(
            semanticscholar_id
        )

        return DocumentIdentifier(
            d3_document_id=d3_document_id,
            semanticscholar_id=semanticscholar_id,
            semanticscholar_url=self.input_converter.get_semanticscholar_url_from_d3_document_id(
                d3_document_id
            ),
            arxiv_id=self.input_converter.get_arxiv_id_from_d3_document_id(d3_document_id),
            arxiv_url=self.input_converter.get_arxiv_url_from_d3_document_id(d3_document_id),
        )

    def get_identifier_from_semanticscholar_url(
        self, semanticscholar_url: str
    ) -> DocumentIdentifier:
        d3_document_id = self.input_converter.get_d3_document_id_from_semanticscholar_url(
            semanticscholar_url
        )

        return DocumentIdentifier(
            d3_document_id=d3_document_id,
            semanticscholar_id=self.input_converter.get_semanticscholar_id_from_d3_document_id(
                d3_document_id
            ),
            semanticscholar_url=semanticscholar_url,
            arxiv_id=self.input_converter.get_arxiv_id_from_d3_document_id(d3_document_id),
            arxiv_url=self.input_converter.get_arxiv_url_from_d3_document_id(d3_document_id),
        )

    def get_identifier_from_arxiv_id(self, arxiv_id: str) -> DocumentIdentifier:
        d3_document_id = self.input_converter.get_d3_document_id_from_arxiv_id(arxiv_id)

        return DocumentIdentifier(
            d3_document_id=d3_document_id,
            semanticscholar_id=self.input_converter.get_semanticscholar_id_from_d3_document_id(
                d3_document_id
            ),
            semanticscholar_url=self.input_converter.get_semanticscholar_url_from_d3_document_id(
                d3_document_id
            ),
            arxiv_id=arxiv_id,
            arxiv_url=self.input_converter.get_arxiv_url_from_d3_document_id(d3_document_id),
        )

    def get_identifier_from_arxiv_url(self, arxiv_url: str) -> DocumentIdentifier:
        d3_document_id = self.input_converter.get_d3_document_id_from_arxiv_url(arxiv_url)

        return DocumentIdentifier(
            d3_document_id=d3_document_id,
            semanticscholar_id=self.input_converter.get_semanticscholar_id_from_d3_document_id(
                d3_document_id
            ),
            semanticscholar_url=self.input_converter.get_semanticscholar_url_from_d3_document_id(
                d3_document_id
            ),
            arxiv_id=self.input_converter.get_arxiv_id_from_d3_document_id(d3_document_id),
            arxiv_url=arxiv_url,
        )

    def get_co_citation_analysis_scores(self) -> ScoresFrame:
        return read_df_from_parquet(
            ResultsPaths.citation_models.co_citation_analysis_scores_parquet
        )

    def get_bibliographic_coupling_scores(self) -> ScoresFrame:
        return read_df_from_parquet(
            ResultsPaths.citation_models.bibliographic_coupling_scores_parquet
        )

    def get_citation_model_data(self) -> CitationModelData:
        assert self.identifier.d3_document_id is not None

        citation_model_data_constructor = CitationModelDataConstructor(
            d3_document_id=self.identifier.d3_document_id,
            documents_frame=self.documents_frame,
            constructor_plugin=self.model_data_constructor_plugin,
            co_citation_analysis_scores_frame=self.get_co_citation_analysis_scores(),
            bibliographic_coupling_scores_frame=self.get_bibliographic_coupling_scores(),
        )
        return CitationModelData.from_constructor(citation_model_data_constructor)

    def get_cosine_similarities(self) -> ScoresFrame:
        return load_cosine_similarities_from_choice(self.language_model_choice)

    def get_language_model_data(self) -> LanguageModelData:
        assert self.identifier.d3_document_id is not None

        language_model_data_constructor = LanguageModelDataConstructor(
            d3_document_id=self.identifier.d3_document_id,
            documents_frame=self.documents_frame,
            constructor_plugin=self.model_data_constructor_plugin,
            cosine_similarity_scores_frame=self.get_cosine_similarities(),
        )
        return LanguageModelData.from_constructor(language_model_data_constructor)
