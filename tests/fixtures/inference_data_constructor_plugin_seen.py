import polars as pl
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference.constructor_plugin_seen import SeenInferenceDataConstructorPlugin
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils.aliases import DocumentsFrame, ScoresFrame


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_semanticscholar_id(
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
def seen_paper_attribute_getter_from_semanticscholar_url(
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
def seen_paper_attribute_getter_from_arxiv_id(
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
def seen_paper_attribute_getter_from_arxiv_url(
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


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_co_citation_analysis(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_co_citation_analysis_scores()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_bibliographic_coupling(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_bibliographic_coupling_scores()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_tfidf(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_bm25(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.BM25,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_word2vec(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.WORD2VEC,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_glove(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.GLOVE,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_fasttext(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.FASTTEXT,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_bert(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.BERT,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_scibert(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.SCIBERT,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_longformer(
    test_documents_frame: DocumentsFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.LONGFORMER,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data(
    test_documents_frame: DocumentsFrame,
) -> CitationModelData:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_citation_model_data()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data_query_document(
    seen_paper_attribute_getter_citation_model_data: CitationModelData,
) -> DocumentInfo:
    return seen_paper_attribute_getter_citation_model_data.query_document


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data_integer_labels(
    seen_paper_attribute_getter_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_citation_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data_info_matrix(
    seen_paper_attribute_getter_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_citation_model_data.info_frame


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data_feature_matrix(
    seen_paper_attribute_getter_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_citation_model_data.features_frame


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data(
    test_documents_frame: DocumentsFrame,
) -> LanguageModelData:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenInferenceDataConstructorPlugin(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
        documents_frame=test_documents_frame,
    )

    return seen_paper_attribute_getter.get_language_model_data()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data_query_document(
    seen_paper_attribute_getter_language_model_data: LanguageModelData,
) -> DocumentInfo:
    return seen_paper_attribute_getter_language_model_data.query_document


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data_integer_labels(
    seen_paper_attribute_getter_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_language_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data_info_matrix(
    seen_paper_attribute_getter_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_language_model_data.info_frame
