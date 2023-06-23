import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import (
    SeenInferenceDataConstructorPlugin,
    UnseenInferenceDataConstructorPlugin,
)
from readnext.inference.document_identifier import DocumentIdentifier
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


# SECTION: Constructor Fixtures
# SUBSECTION: Seen Constructor Plugin
@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_from_semanticscholar_id(
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
def inference_data_constructor_plugin_seen_from_semanticscholar_url(
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
def inference_data_constructor_plugin_seen_from_arxiv_id(
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
def inference_data_constructor_plugin_seen_from_arxiv_url(
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
def inference_data_constructor_plugin_unseen_from_semanticscholar_id(
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
def inference_data_constructor_plugin_unseen_from_semanticscholar_url(
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
def inference_data_constructor_plugin_unseen_from_arxiv_id(
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
def inference_data_constructor_plugin_unseen_from_arxiv_url(
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


# SECTION: Document Identifier Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_document_identifier_from_semanticscholar_id(
    inference_data_constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> DocumentIdentifier:
    return inference_data_constructor_plugin_seen.identifier


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_document_identifier_from_semanticscholar_url(
    inference_data_constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> DocumentIdentifier:
    return inference_data_constructor_plugin_seen.identifier


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_document_identifier_from_arxiv_id(
    inference_data_constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> DocumentIdentifier:
    return inference_data_constructor_plugin_seen.identifier


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_document_identifier_from_arxiv_url(
    inference_data_constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> DocumentIdentifier:
    return inference_data_constructor_plugin_seen.identifier


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_document_identifier_from_semanticscholar_id(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> DocumentIdentifier:
    return inference_data_constructor_plugin_unseen.identifier


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_document_identifier_from_semanticscholar_url(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> DocumentIdentifier:
    return inference_data_constructor_plugin_unseen.identifier


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_document_identifier_from_arxiv_id(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> DocumentIdentifier:
    return inference_data_constructor_plugin_unseen.identifier


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_document_identifier_from_arxiv_url(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> DocumentIdentifier:
    return inference_data_constructor_plugin_unseen.identifier


# SECTION: Constructor Plugin Methods
@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen(
    inference_data_constructor_plugin_seen_from_arxiv_url: SeenInferenceDataConstructorPlugin,
) -> SeenInferenceDataConstructorPlugin:
    return inference_data_constructor_plugin_seen_from_arxiv_url


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen(
    inference_data_constructor_plugin_unseen_from_arxiv_url: SeenInferenceDataConstructorPlugin,
) -> SeenInferenceDataConstructorPlugin:
    return inference_data_constructor_plugin_unseen_from_arxiv_url


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


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_co_citation_analysis(
    inference_data_constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> ScoresFrame:
    return inference_data_constructor_plugin_seen.get_co_citation_analysis_scores()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_co_citation_analysis(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> ScoresFrame:
    return inference_data_constructor_plugin_unseen.get_co_citation_analysis_scores()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_bibliographic_coupling(
    inference_data_constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> ScoresFrame:
    return inference_data_constructor_plugin_seen.get_bibliographic_coupling_scores()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_bibliographic_coupling(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> ScoresFrame:
    return inference_data_constructor_plugin_unseen.get_bibliographic_coupling_scores()


@pytest.fixture(scope="session", params=language_model_choices)
def inference_data_constructor_plugin_seen_cosine_similarities(
    request: pytest.FixtureRequest,
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    """
    Fixture for cosine similarities of all language models for seen papers (fast).
    """
    arxiv_url = "https://arxiv.org/abs/1706.03762"

    inference_data_constructor_plugin = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=arxiv_url,
        language_model_choice=request.param,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return inference_data_constructor_plugin.get_cosine_similarities()


# test only a single language model for unseen papers due to computational time
@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_cosine_similarities(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> ScoresFrame:
    return inference_data_constructor_plugin_unseen.get_cosine_similarities()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_citation_model_data(
    inference_data_constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> CitationModelData:
    return inference_data_constructor_plugin_seen.get_citation_model_data()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_citation_model_data(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> CitationModelData:
    return inference_data_constructor_plugin_unseen.get_citation_model_data()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_language_model_data(
    inference_data_constructor_plugin_seen: SeenInferenceDataConstructorPlugin,
) -> LanguageModelData:
    return inference_data_constructor_plugin_seen.get_language_model_data()


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_language_model_data(
    inference_data_constructor_plugin_unseen: UnseenInferenceDataConstructorPlugin,
) -> LanguageModelData:
    return inference_data_constructor_plugin_unseen.get_language_model_data()


# SECTION: Model Data Attributes
model_data_pair_seen = [
    lazy_fixture("inference_data_constructor_plugin_seen_citation_model_data"),
    lazy_fixture("inference_data_constructor_plugin_seen_language_model_data"),
]

model_data_pair_unseen = [
    lazy_fixture("inference_data_constructor_plugin_unseen_citation_model_data"),
    lazy_fixture("inference_data_constructor_plugin_unseen_language_model_data"),
]


@pytest.fixture(scope="session", params=model_data_pair_seen)
def inference_data_constructor_plugin_seen_model_data_query_document(
    request: pytest.FixtureRequest,
) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=model_data_pair_unseen)
def inference_data_constructor_plugin_unseen_model_data_query_document(
    request: pytest.FixtureRequest,
) -> DocumentInfo:
    return request.param.query_document


@pytest.fixture(scope="session", params=model_data_pair_seen)
def inference_data_constructor_plugin_seen_model_data_info_frame(
    request: pytest.FixtureRequest,
) -> InfoFrame:
    return request.param.info_frame


@pytest.fixture(scope="session", params=model_data_pair_unseen)
def inference_data_constructor_plugin_unseen_model_data_info_frame(
    request: pytest.FixtureRequest,
) -> InfoFrame:
    return request.param.info_frame


@pytest.fixture(scope="session", params=model_data_pair_seen)
def inference_data_constructor_plugin_seen_model_data_integer_labels(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session", params=model_data_pair_unseen)
def inference_data_constructor_plugin_unseen_model_data_integer_labels(
    request: pytest.FixtureRequest,
) -> IntegerLabelsFrame:
    return request.param.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_citation_model_data_features_frame(
    inference_data_constructor_plugin_seen_citation_model_data: CitationModelData,
) -> CitationFeaturesFrame:
    return inference_data_constructor_plugin_seen_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_citation_model_data_features_frame(
    inference_data_constructor_plugin_unseen_citation_model_data: CitationModelData,
) -> CitationFeaturesFrame:
    return inference_data_constructor_plugin_unseen_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_language_model_data_features_frame(
    inference_data_constructor_plugin_seen_language_model_data: LanguageModelData,
) -> LanguageFeaturesFrame:
    return inference_data_constructor_plugin_seen_language_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_language_model_data_features_frame(
    inference_data_constructor_plugin_unseen_language_model_data: LanguageModelData,
) -> LanguageFeaturesFrame:
    return inference_data_constructor_plugin_unseen_language_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_citation_model_data_ranks_frame(
    inference_data_constructor_plugin_seen_citation_model_data: CitationModelData,
) -> CitationRanksFrame:
    return inference_data_constructor_plugin_seen_citation_model_data.ranks_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_citation_model_data_ranks_frame(
    inference_data_constructor_plugin_unseen_citation_model_data: CitationModelData,
) -> CitationRanksFrame:
    return inference_data_constructor_plugin_unseen_citation_model_data.ranks_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_seen_citation_model_data_points_frame(
    inference_data_constructor_plugin_seen_citation_model_data: CitationModelData,
) -> CitationPointsFrame:
    return inference_data_constructor_plugin_seen_citation_model_data.points_frame


@pytest.fixture(scope="session")
def inference_data_constructor_plugin_unseen_citation_model_data_points_frame(
    inference_data_constructor_plugin_unseen_citation_model_data: CitationModelData,
) -> CitationPointsFrame:
    return inference_data_constructor_plugin_unseen_citation_model_data.points_frame
