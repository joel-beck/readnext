import pytest
from pytest_lazyfixture import lazy_fixture
from pydantic import ValidationError
from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import InferenceDataConstructor
from readnext.inference.constructor_plugin_seen import SeenInferenceDataConstructorPlugin
from readnext.inference.constructor_plugin_unseen import UnseenInferenceDataConstructorPlugin
from readnext.modeling.language_models import LanguageModelChoice

seen_inference_data_constructors_slow_skip_ci = [
    "inference_data_constructor_seen_from_semanticscholar_id"
]
unseen_inference_data_constructors_skip_ci = ["inference_data_constructor_unseen_from_arxiv_url"]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "inference_data_constructor", lazy_fixture(seen_inference_data_constructors_slow_skip_ci)
)
def test_seen_attribute_getter_is_selected_correctly(
    inference_data_constructor: InferenceDataConstructor,
) -> None:
    assert inference_data_constructor.query_document_in_training_data()
    assert isinstance(
        inference_data_constructor.constructor_plugin,
        SeenInferenceDataConstructorPlugin,
    )


@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "inference_data_constructor", lazy_fixture(unseen_inference_data_constructors_skip_ci)
)
def test_unseen_attribute_getter_is_selected_correctly(
    inference_data_constructor: InferenceDataConstructor,
) -> None:
    assert not inference_data_constructor.query_document_in_training_data()
    assert isinstance(
        inference_data_constructor.constructor_plugin,
        UnseenInferenceDataConstructorPlugin,
    )



def test_pydantic_validation() -> None:
    # Test valid data
    data = {
        "semanticscholar_id": "1234567890123456789012345678901234567890",  # 40 chars long
        "language_model_choice": LanguageModelChoice.TFIDF,
        "feature_weights": FeatureWeights(co_citation_analysis_weight=1.0, ...)  # Continue with other fields
    }
    instance = InferenceDataConstructor(**data)
    assert instance.semanticscholar_id == "1234567890123456789012345678901234567890"
    assert instance.language_model_choice == LanguageModelChoice.TFIDF

    # Test missing required data
    with pytest.raises(ValidationError) as exc_info:
        instance = InferenceDataConstructor()
    assert "value_error.missing" in str(exc_info.value)

    # Test semanticscholar_id too short
    data = {
        "semanticscholar_id": "1234567890",  # Too short
        "language_model_choice": LanguageModelChoice.TFIDF,
        "feature_weights": FeatureWeights(co_citation_analysis_weight=1.0, ...)  # Continue with other fields
    }
    with pytest.raises(ValidationError) as exc_info:
        instance = InferenceDataConstructor(**data)
    assert "value_error.any_str.min_length" in str(exc_info.value)

    # Test semanticscholar_id too long
    data = {
        "semanticscholar_id": "12345678901234567890123456789012345678901234567890",  # Too long
        "language_model_choice": LanguageModelChoice.TFIDF,
        "feature_weights": FeatureWeights(co_citation_analysis_weight=1.0, ...)  # Continue with other fields
    }
    with pytest.raises(ValidationError) as exc_info:
        instance = InferenceDataConstructor(**data)
    assert "value_error.any_str.max_length" in str(exc_info.value)

    # Test wrong semanticscholar_url
    data = {
        "semanticscholar_url": "https://wrongurl.com/paper/",
        "language_model_choice": LanguageModelChoice.TFIDF,
        "feature_weights": FeatureWeights(co_citation_analysis_weight=1.0, ...)  # Continue with other fields
    }
    with pytest.raises(ValueError) as exc_info:
        instance = InferenceDataConstructor(**data)
    assert "Semanticscholar URL must start with `https://www.semanticscholar.org/paper/`" in str(exc_info.value)

    # Test wrong arxiv_url
    data = {
        "arxiv_url": "https://wrongurl.com/abs/",
        "language_model_choice": LanguageModelChoice.TFIDF,
        "feature_weights": FeatureWeights(co_citation_analysis_weight=1.0, ...)  # Continue with other fields
    }
    with pytest.raises(ValueError) as exc_info:
        instance = InferenceDataConstructor(**data)
    assert "Arxiv URL must start with `https://arxiv.org/abs/`" in str(exc_info.value)

    # Test all URL and ID missing
    data = {
        "language_model_choice": LanguageModelChoice.TFIDF,
        "feature_weights": FeatureWeights(co_citation_analysis_weight=1.0, ...)  # Continue with other fields
    }
    with pytest.raises(ValueError) as exc_info:
        instance = InferenceDataConstructor(**data)
    assert "At least one of `semanticscholar_id`, `semanticscholar_url`, `arxiv_url` must be provided." in str(exc_info.value)



def test_kw_only_initialization_inference_data_constructor() -> None:
    with pytest.raises(TypeError):
        InferenceDataConstructor(
            "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",  # type: ignore
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "1905.12616",
            "https://arxiv.org/abs/1905.12616",
            LanguageModelChoice.TFIDF,
            FeatureWeights(),
        )
