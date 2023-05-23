import pandas as pd
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import (
    DocumentIdentifier,
    InferenceData,
    InferenceDataConstructor,
    InferenceDataInputConverter,
    LanguageModelChoice,
)
from readnext.inference.attribute_getter import SeenPaperAttributeGetter, UnseenPaperAttributeGetter

# These imports must not come from `readnext.inference`, otherwise they are really
# imported twice with different session scopes and `isinstance()` checks fail.
from readnext.inference.inference_data_constructor import Features, Labels, Ranks, Recommendations
from readnext.modeling import DocumentInfo
from readnext.modeling.language_models.model_choice import LanguageModelChoice


# SECTION: Input Converter
@pytest.fixture
def toy_data() -> pd.DataFrame:
    index = pd.Index([1001, 1002], name="document_id", dtype=pd.Int64Dtype())
    return pd.DataFrame(
        {
            "semanticscholar_url": [
                "https://www.semanticscholar.org/paper/1",
                "https://www.semanticscholar.org/paper/2",
            ],
            "arxiv_id": ["1", "2"],
        },
        index=index,
    )


@pytest.fixture
def input_converter_toy_data(toy_data: pd.DataFrame) -> InferenceDataInputConverter:
    return InferenceDataInputConverter(toy_data)


# SECTION: Seen Attribute Getter
@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_semanticscholar_id(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> SeenPaperAttributeGetter:
    # NOTE: Make sure the test document is also seen in the TEST documents data, i.e. within
    # the e.g. top 100 most cited papers
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    return SeenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_semanticscholar_url(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> SeenPaperAttributeGetter:
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )

    return SeenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=semanticscholar_url,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_arxiv_id(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> SeenPaperAttributeGetter:
    arxiv_id = "1706.03762"

    return SeenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=arxiv_id,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_arxiv_url(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> SeenPaperAttributeGetter:
    arxiv_url = "https://arxiv.org/abs/1706.03762"

    return SeenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


# SECTION: Unseen Attribute Getter
@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_from_semanticscholar_id(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> UnseenPaperAttributeGetter:
    semanticscholar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    return UnseenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_from_semanticscholar_url(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> UnseenPaperAttributeGetter:
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    )

    return UnseenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=semanticscholar_url,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_from_arxiv_id(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> UnseenPaperAttributeGetter:
    arxiv_id = "2303.08774"

    return UnseenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=arxiv_id,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_from_arxiv_url(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> UnseenPaperAttributeGetter:
    arxiv_url = "https://arxiv.org/abs/2303.08774"

    return UnseenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


# SECTION: Inference Data Constructor
@pytest.fixture(scope="session")
def inference_data_seen_constructor_from_semanticscholar_id() -> InferenceDataConstructor:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    return InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )


@pytest.fixture(scope="session")
def inference_data_unseen_constructor_from_arxiv_url() -> InferenceDataConstructor:
    arxiv_url = "https://arxiv.org/abs/2303.08774"
    return InferenceDataConstructor(
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.scibert,
        feature_weights=FeatureWeights(),
    )


# SECTION: Inference Data
@pytest.fixture(scope="session")
def inference_data_seen_from_semanticscholar_id(
    inference_data_seen_constructor_from_semanticscholar_id: InferenceDataConstructor,
) -> InferenceData:
    return InferenceData.from_constructor(inference_data_seen_constructor_from_semanticscholar_id)


@pytest.fixture(scope="session")
def inference_data_unseen_from_arxiv_url(
    inference_data_unseen_constructor_from_arxiv_url: InferenceDataConstructor,
) -> InferenceData:
    return InferenceData.from_constructor(inference_data_unseen_constructor_from_arxiv_url)


@pytest.fixture(scope="session")
def inference_data_seen_document_identifier(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> DocumentIdentifier:
    return inference_data_seen_from_semanticscholar_id.document_identifier


@pytest.fixture(scope="session")
def inference_data_unseen_document_identifier(
    inference_data_unseen_from_arxiv_url: InferenceData,
) -> DocumentIdentifier:
    return inference_data_unseen_from_arxiv_url.document_identifier


@pytest.fixture(scope="session")
def inference_data_seen_document_info(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> DocumentInfo:
    return inference_data_seen_from_semanticscholar_id.document_info


@pytest.fixture(scope="session")
def inference_data_unseen_document_info(
    inference_data_unseen_from_arxiv_url: InferenceData,
) -> DocumentInfo:
    return inference_data_unseen_from_arxiv_url.document_info


@pytest.fixture(scope="session")
def inference_data_seen_features(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> Features:
    return inference_data_seen_from_semanticscholar_id.features


@pytest.fixture(scope="session")
def inference_data_unseen_features(inference_data_unseen_from_arxiv_url: InferenceData) -> Features:
    return inference_data_unseen_from_arxiv_url.features


@pytest.fixture(scope="session")
def inference_data_seen_ranks(inference_data_seen_from_semanticscholar_id: InferenceData) -> Ranks:
    return inference_data_seen_from_semanticscholar_id.ranks


@pytest.fixture(scope="session")
def inference_data_unseen_ranks(inference_data_unseen_from_arxiv_url: InferenceData) -> Ranks:
    return inference_data_unseen_from_arxiv_url.ranks


@pytest.fixture(scope="session")
def inference_data_seen_labels(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> Labels:
    return inference_data_seen_from_semanticscholar_id.labels


@pytest.fixture(scope="session")
def inference_data_unseen_labels(inference_data_unseen_from_arxiv_url: InferenceData) -> Labels:
    return inference_data_unseen_from_arxiv_url.labels


@pytest.fixture(scope="session")
def inference_data_seen_recommendations(
    inference_data_seen_from_semanticscholar_id: InferenceData,
) -> Recommendations:
    return inference_data_seen_from_semanticscholar_id.recommendations


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_citation_to_language(
    inference_data_seen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_seen_recommendations.citation_to_language


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_citation_to_language_candidates(
    inference_data_seen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_seen_recommendations.citation_to_language_candidates


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_language_to_citation(
    inference_data_seen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_seen_recommendations.language_to_citation


@pytest.fixture(scope="session")
def inference_data_seen_recommendations_language_to_citation_candidates(
    inference_data_seen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_seen_recommendations.language_to_citation_candidates


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations(
    inference_data_unseen_from_arxiv_url: InferenceData,
) -> Recommendations:
    return inference_data_unseen_from_arxiv_url.recommendations


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_citation_to_language(
    inference_data_unseen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_unseen_recommendations.citation_to_language


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_citation_to_language_candidates(
    inference_data_unseen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_unseen_recommendations.citation_to_language_candidates


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_language_to_citation(
    inference_data_unseen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_unseen_recommendations.language_to_citation


@pytest.fixture(scope="session")
def inference_data_unseen_recommendations_language_to_citation_candidates(
    inference_data_unseen_recommendations: Recommendations,
) -> pd.DataFrame:
    return inference_data_unseen_recommendations.language_to_citation_candidates
