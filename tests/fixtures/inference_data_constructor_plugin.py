import polars as pl
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import SeenInferenceDataConstructorPlugin, InferenceDataConstructorPlugin
from readnext.inference import UnseenInferenceDataConstructorPlugin
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils.aliases import DocumentsFrame, ScoresFrame

semanticscholar_id_plugin_pairs = [
    ("204e3073870fae3d05bcbc2f6a8e263d9b72e776", SeenInferenceDataConstructorPlugin),
    ("8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48", UnseenInferenceDataConstructorPlugin),
]

language_model_choices = [
    LanguageModelChoice.TFIDF,
    LanguageModelChoice.BM25,
    LanguageModelChoice.WORD2VEC,
    LanguageModelChoice.GLOVE,
    LanguageModelChoice.FASTTEXT,
    LanguageModelChoice.BERT,
    LanguageModelChoice.SCIBERT,
    LanguageModelChoice.LONGFORMER,
]


id_plugin_model_choice_triples = [
    (id_plugin, model_choice)
    for id_plugin in semanticscholar_id_plugin_pairs
    for model_choice in language_model_choices
]


# SECTION: Constructor Fixtures
# SUBSECTION: Seen Constructor Plugin
@pytest.fixture(scope="session")
def seen_inference_data_constructor_plugin_from_semanticscholar_id(
    test_documents_frame: DocumentsFrame,
) -> SeenInferenceDataConstructorPlugin:
    # Make sure the test document is also seen in the TEST documents data, i.e. within
    # the e.g. top 100 most cited papers!
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    return SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


@pytest.fixture(scope="session")
def seen_inference_data_constructor_plugin_from_semanticscholar_url(
    test_documents_frame: DocumentsFrame,
) -> SeenInferenceDataConstructorPlugin:
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )

    return SeenInferenceDataConstructorPlugin(
        semanticscholar_id=None,
        semanticscholar_url=semanticscholar_url,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


@pytest.fixture(scope="session")
def seen_inference_data_constructor_plugin_from_arxiv_id(
    test_documents_frame: DocumentsFrame,
) -> SeenInferenceDataConstructorPlugin:
    arxiv_id = "1706.03762"

    return SeenInferenceDataConstructorPlugin(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=arxiv_id,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


@pytest.fixture(scope="session")
def seen_inference_data_constructor_plugin_from_arxiv_url(
    test_documents_frame: DocumentsFrame,
) -> SeenInferenceDataConstructorPlugin:
    arxiv_url = "https://arxiv.org/abs/1706.03762"

    return SeenInferenceDataConstructorPlugin(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


# SUBSECTION: Unseen Constructor Plugin
@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_from_semanticscholar_id(
    test_documents_frame: DocumentsFrame,
) -> UnseenInferenceDataConstructorPlugin:
    semanticscholar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    return UnseenInferenceDataConstructorPlugin(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_from_semanticscholar_url(
    test_documents_frame: DocumentsFrame,
) -> UnseenInferenceDataConstructorPlugin:
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    )

    return UnseenInferenceDataConstructorPlugin(
        semanticscholar_id=None,
        semanticscholar_url=semanticscholar_url,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_from_arxiv_id(
    test_documents_frame: DocumentsFrame,
) -> UnseenInferenceDataConstructorPlugin:
    arxiv_id = "2303.08774"

    return UnseenInferenceDataConstructorPlugin(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=arxiv_id,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_from_arxiv_url(
    test_documents_frame: DocumentsFrame,
) -> UnseenInferenceDataConstructorPlugin:
    arxiv_url = "https://arxiv.org/abs/2303.08774"

    return UnseenInferenceDataConstructorPlugin(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


# SECTION: Constructor Plugin Methods
@pytest.fixture(scope="session", params=semanticscholar_id_plugin_pairs)
def inference_data_constructor_plugin(
    request: pytest.FixtureRequest,
    test_documents_frame: DocumentsFrame,
) -> InferenceDataConstructorPlugin:
    """
    Base `InfereceDataConstructorPlugin` fixture for seen and unseen papers. Considers
    only a single language model and a single set of feature weights.
    """
    semantischolar_id, PluginClass = request.param

    return PluginClass(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_co_citation_analysis(
    inference_data_constructor_plugin: InferenceDataConstructorPlugin,
) -> ScoresFrame:
    return inference_data_constructor_plugin.get_co_citation_analysis_scores()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_bibliographic_coupling(
    inference_data_constructor_plugin: InferenceDataConstructorPlugin,
) -> ScoresFrame:
    return inference_data_constructor_plugin.get_bibliographic_coupling_scores()


@pytest.fixture(scope="session", params=id_plugin_model_choice_triples)
def inference_data_constructor_plugin_cosine_similarities(
    request: pytest.FixtureRequest,
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    """
    Fixture for cosine similarities of all language models.
    """
    (semantischolar_id, PluginClass), language_model_choice = request.param

    inference_data_constructor_plugin = PluginClass(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=language_model_choice,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return inference_data_constructor_plugin.get_cosine_similarities()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_citation_model_data(
    inference_data_constructor_plugin: InferenceDataConstructorPlugin,
) -> CitationModelData:
    return inference_data_constructor_plugin.get_citation_model_data()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_citation_model_data_query_document(
    inference_data_constructor_plugin_citation_model_data: CitationModelData,
) -> DocumentInfo:
    return inference_data_constructor_plugin_citation_model_data.query_document


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_citation_model_data_integer_labels(
    inference_data_constructor_plugin_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_plugin_citation_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_citation_model_data_info_matrix(
    inference_data_constructor_plugin_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_plugin_citation_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_citation_model_data_feature_matrix(
    inference_data_constructor_plugin_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_plugin_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_language_model_data(
    inference_data_constructor_plugin: InferenceDataConstructorPlugin,
) -> LanguageModelData:
    return inference_data_constructor_plugin.get_language_model_data()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_language_model_data_query_document(
    inference_data_constructor_plugin_language_model_data: LanguageModelData,
) -> DocumentInfo:
    return inference_data_constructor_plugin_language_model_data.query_document


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_language_model_data_integer_labels(
    inference_data_constructor_plugin_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_plugin_language_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_language_model_data_info_matrix(
    inference_data_constructor_plugin_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_plugin_language_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_language_model_data_features_frame(
    inference_data_constructor_plugin_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_plugin_language_model_data.features_frame
