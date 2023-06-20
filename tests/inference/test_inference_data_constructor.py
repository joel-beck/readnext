import pytest
from pydantic import ValidationError

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import InferenceDataConstructor
from readnext.inference.constructor_plugin_seen import SeenInferenceDataConstructorPlugin
from readnext.inference.constructor_plugin_unseen import UnseenInferenceDataConstructorPlugin
from readnext.modeling.language_models import LanguageModelChoice


@pytest.mark.updated
@pytest.mark.skip_ci
def test_seen_attribute_getter_is_selected_correctly(
    inference_data_constructor_seen: InferenceDataConstructor,
) -> None:
    assert inference_data_constructor_seen.query_document_in_training_data()
    assert isinstance(
        inference_data_constructor_seen.constructor_plugin,
        SeenInferenceDataConstructorPlugin,
    )


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
def test_unseen_attribute_getter_is_selected_correctly(
    inference_data_constructor_unseen: InferenceDataConstructor,
) -> None:
    assert not inference_data_constructor_unseen.query_document_in_training_data()
    assert isinstance(
        inference_data_constructor_unseen.constructor_plugin,
        UnseenInferenceDataConstructorPlugin,
    )


@pytest.mark.updated
def test_pydantic_validation() -> None:
    # Test valid data
    inference_data_constructor = InferenceDataConstructor(
        semanticscholar_id="1234567890123456789012345678901234567890",
        language_model_choice=LanguageModelChoice.TFIDF,
        feature_weights=FeatureWeights(),
    )
    assert (
        inference_data_constructor.semanticscholar_id == "1234567890123456789012345678901234567890"
    )
    assert inference_data_constructor.language_model_choice == LanguageModelChoice.TFIDF

    # Test missing required data
    with pytest.raises(TypeError):
        InferenceDataConstructor()  # type: ignore

    # Test semanticscholar_id too short
    with pytest.raises(ValidationError):
        InferenceDataConstructor(
            semanticscholar_id="1234567890",
            language_model_choice=LanguageModelChoice.TFIDF,
            feature_weights=FeatureWeights(),
        )

    # Test semanticscholar_id too long
    with pytest.raises(ValidationError):
        InferenceDataConstructor(
            semanticscholar_id="12345678901234567890123456789012345678901234567890",
            language_model_choice=LanguageModelChoice.TFIDF,
            feature_weights=FeatureWeights(),
        )

    # Test wrong semanticscholar_url
    with pytest.raises(ValueError):
        InferenceDataConstructor(
            semanticscholar_url="https://wrongurl.com/paper/",
            language_model_choice=LanguageModelChoice.TFIDF,
            feature_weights=FeatureWeights(),
        )

    # Test wrong arxiv_url
    with pytest.raises(ValueError):
        InferenceDataConstructor(
            arxiv_url="https://wrongurl.com/abs/",
            language_model_choice=LanguageModelChoice.TFIDF,
            feature_weights=FeatureWeights(),
        )

    # Test all URL and ID missing
    with pytest.raises(ValueError):
        InferenceDataConstructor(
            language_model_choice=LanguageModelChoice.TFIDF,
            feature_weights=FeatureWeights(),
        )


@pytest.mark.updated
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
