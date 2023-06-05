import polars as pl
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import (
    DocumentIdentifier,
    Features,
    InferenceDataConstructor,
    Labels,
    Ranks,
    Recommendations,
)
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils import ScoresFrame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_from_arxiv_url() -> InferenceDataConstructor:
    arxiv_url = "https://arxiv.org/abs/2303.08774"
    return InferenceDataConstructor(
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.scibert,
        feature_weights=FeatureWeights(),
    )


# SECTION: Attribute Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_unseen_documents_data(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> pl.DataFrame:
    return inference_data_constructor_unseen_from_arxiv_url._documents_data


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_co_citation_analysis_scores(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_unseen_from_arxiv_url._co_citation_analysis_scores


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_bibliographic_coupling_scores(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_unseen_from_arxiv_url._bibliographic_coupling_scores


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_cosine_similarities(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_unseen_from_arxiv_url._cosine_similarities


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> CitationModelData:
    return inference_data_constructor_unseen_from_arxiv_url._citation_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data_query_document(
    inference_data_constructor_unseen_citation_model_data: CitationModelData,
) -> DocumentInfo:
    return inference_data_constructor_unseen_citation_model_data.query_document


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data_integer_labels(
    inference_data_constructor_unseen_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_unseen_citation_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data_info_matrix(
    inference_data_constructor_unseen_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_unseen_citation_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_citation_model_data_feature_matrix(
    inference_data_constructor_unseen_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_unseen_citation_model_data.features_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_language_model_data(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> LanguageModelData:
    return inference_data_constructor_unseen_from_arxiv_url._language_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_language_model_data_query_document(
    inference_data_constructor_unseen_language_model_data: LanguageModelData,
) -> DocumentInfo:
    return inference_data_constructor_unseen_language_model_data.query_document


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_language_model_data_integer_labels(
    inference_data_constructor_unseen_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_unseen_language_model_data.integer_labels_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_language_model_data_info_matrix(
    inference_data_constructor_unseen_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_unseen_language_model_data.info_frame


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_language_model_data_cosine_similarity_ranks(
    inference_data_constructor_unseen_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_unseen_language_model_data.cosine_similarity_ranks


# SECTION: Method Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_unseen_document_identifier(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> DocumentIdentifier:
    return inference_data_constructor_unseen_from_arxiv_url.collect_document_identifier()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_document_info(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> DocumentInfo:
    return inference_data_constructor_unseen_from_arxiv_url.collect_document_info()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_features(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> Features:
    return inference_data_constructor_unseen_from_arxiv_url.collect_features()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_ranks(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> Ranks:
    return inference_data_constructor_unseen_from_arxiv_url.collect_ranks()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_labels(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> Labels:
    return inference_data_constructor_unseen_from_arxiv_url.collect_labels()


@pytest.fixture(scope="session")
def inference_data_constructor_unseen_recommendations(
    inference_data_constructor_unseen_from_arxiv_url: InferenceDataConstructor,
) -> Recommendations:
    return inference_data_constructor_unseen_from_arxiv_url.collect_recommendations()
