import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import (
    SeenInferenceDataConstructorPlugin,
    UnseenInferenceDataConstructorPlugin,
)
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils.aliases import (
    CitationFeaturesFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    DocumentsFrame,
    InfoFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
    ScoresFrame,
)


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

# id_plugin_model_choice_triples = [
#     (id_plugin, model_choice)
#     for id_plugin in semanticscholar_id_plugin_pairs
#     for model_choice in language_model_choices
# ]


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
def unseen_inference_data_constructor_plugin_from_semanticscholar_id(
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
def unseen_inference_data_constructor_plugin_from_semanticscholar_url(
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
def unseen_inference_data_constructor_plugin_from_arxiv_id(
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
def unseen_inference_data_constructor_plugin_from_arxiv_url(
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
@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("seen_inference_data_constructor_plugin_from_arxiv_url"),
        lazy_fixture("unseen_inference_data_constructor_plugin_from_arxiv_url"),
    ],
)
def inference_data_constructor_plugin_co_citation_analysis(
    request: pytest.FixtureRequest,
) -> ScoresFrame:
    return request.param.get_co_citation_analysis_scores()


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("seen_inference_data_constructor_plugin_from_arxiv_url"),
        lazy_fixture("unseen_inference_data_constructor_plugin_from_arxiv_url"),
    ],
)
def inference_data_constructor_plugin_bibliographic_coupling(
    request: pytest.FixtureRequest,
) -> ScoresFrame:
    return request.param.get_bibliographic_coupling_scores()


# TODO:
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


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("seen_inference_data_constructor_plugin_from_arxiv_url"),
        lazy_fixture("unseen_inference_data_constructor_plugin_from_arxiv_url"),
    ],
)
def inference_data_constructor_plugin_citation_model_data(
    request: pytest.FixtureRequest,
) -> CitationModelData:
    return request.param.get_citation_model_data()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_language_model_data(
    request: pytest.FixtureRequest,
) -> LanguageModelData:
    return request.param.get_language_model_data()


# SECTION: Model Data Attributes
@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("inference_data_constructor_plugin_citation_model_data"),
        lazy_fixture("inference_data_constructor_plugin_language_model_data"),
    ],
)
def inference_data_constructor_plugin_model_data_query_document(
    request: pytest.FixtureRequest,
) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("inference_data_constructor_plugin_citation_model_data"),
        lazy_fixture("inference_data_constructor_plugin_language_model_data"),
    ],
)
def inference_data_constructor_plugin_model_data_info_frame(
    request: pytest.FixtureRequest,
) -> InfoFrame:
    return request.param.info_frame


@pytest.fixture(
    scope="session",
    params=[
        lazy_fixture("inference_data_constructor_plugin_citation_model_data"),
        lazy_fixture("inference_data_constructor_plugin_language_model_data"),
    ],
)
def inference_data_constructor_plugin_model_data_integer_labels(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_citation_model_data_features_frame(
    inference_data_constructor_plugin_citation_model_data: CitationModelData,
) -> CitationFeaturesFrame:
    return inference_data_constructor_plugin_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_language_model_data_features_frame(
    inference_data_constructor_plugin_language_model_data: LanguageModelData,
) -> LanguageFeaturesFrame:
    return inference_data_constructor_plugin_language_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_citation_model_data_ranks_frame(
    inference_data_constructor_plugin_citation_model_data: CitationModelData,
) -> CitationRanksFrame:
    return inference_data_constructor_plugin_citation_model_data.ranks_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_citation_model_data_points_frame(
    inference_data_constructor_plugin_citation_model_data: CitationModelData,
) -> CitationPointsFrame:
    return inference_data_constructor_plugin_citation_model_data.points_frame
