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
def inference_data_constructor_seen_from_semanticscholar_id() -> InferenceDataConstructor:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    return InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )


# SECTION: Attribute Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_seen_documents_data(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> pl.DataFrame:
    return inference_data_constructor_seen_from_semanticscholar_id._documents_data


@pytest.fixture(scope="session")
def inference_data_constructor_seen_co_citation_analysis_scores(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_seen_from_semanticscholar_id._co_citation_analysis_scores


@pytest.fixture(scope="session")
def inference_data_constructor_seen_bibliographic_coupling_scores(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_seen_from_semanticscholar_id._bibliographic_coupling_scores


@pytest.fixture(scope="session")
def inference_data_constructor_seen_cosine_similarities(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> ScoresFrame:
    return inference_data_constructor_seen_from_semanticscholar_id._cosine_similarities


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> CitationModelData:
    return inference_data_constructor_seen_from_semanticscholar_id._citation_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_query_document(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> DocumentInfo:
    return inference_data_constructor_seen_citation_model_data.query_document


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_integer_labels(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_seen_citation_model_data.integer_labels


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_info_matrix(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_seen_citation_model_data.info_matrix


@pytest.fixture(scope="session")
def inference_data_constructor_seen_citation_model_data_feature_matrix(
    inference_data_constructor_seen_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return inference_data_constructor_seen_citation_model_data.feature_matrix


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> LanguageModelData:
    return inference_data_constructor_seen_from_semanticscholar_id._language_model_data


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_query_document(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> DocumentInfo:
    return inference_data_constructor_seen_language_model_data.query_document


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_integer_labels(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_seen_language_model_data.integer_labels


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_info_matrix(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_seen_language_model_data.info_matrix


@pytest.fixture(scope="session")
def inference_data_constructor_seen_language_model_data_cosine_similarity_ranks(
    inference_data_constructor_seen_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return inference_data_constructor_seen_language_model_data.cosine_similarity_ranks


# SECTION: Method Fixtures
@pytest.fixture(scope="session")
def inference_data_constructor_seen_document_identifier(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> DocumentIdentifier:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_document_identifier()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_document_info(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> DocumentInfo:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_document_info()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_features(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Features:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_features()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_ranks(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Ranks:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_ranks()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_labels(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Labels:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_labels()


@pytest.fixture(scope="session")
def inference_data_constructor_seen_recommendations(
    inference_data_constructor_seen_from_semanticscholar_id: InferenceDataConstructor,
) -> Recommendations:
    return inference_data_constructor_seen_from_semanticscholar_id.collect_recommendations()
