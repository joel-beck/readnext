import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, InferenceDataConstructor
from readnext.inference.constructor_plugin_seen import SeenInferenceDataConstructorPlugin
from readnext.inference.constructor_plugin_unseen import UnseenInferenceDataConstructorPlugin
from readnext.inference.inference_data_constructor import Features, Labels, Ranks, Recommendations
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


def test_kw_only_initialization_document_identifier() -> None:
    with pytest.raises(TypeError):
        DocumentIdentifier(
            -1,  # type: ignore
            "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "2303.08774",
            "https://arxiv.org/abs/2303.08774",
        )


def test_kw_only_initialization_features() -> None:
    with pytest.raises(TypeError):
        Features(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            FeatureWeights(),
        )


def test_kw_only_initialization_ranks() -> None:
    with pytest.raises(TypeError):
        Ranks(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )


def test_kw_only_initialization_labels() -> None:
    with pytest.raises(TypeError):
        Labels(pl.DataFrame(), pl.DataFrame())  # type: ignore


def test_kw_only_initialization_recommendations() -> None:
    with pytest.raises(TypeError):
        Recommendations(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )


def test_kw_only_initialization_inference_data_constructor() -> None:
    with pytest.raises(TypeError):
        InferenceDataConstructor(
            "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",  # type: ignore
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "1905.12616",
            "https://arxiv.org/abs/1905.12616",
            LanguageModelChoice.tfidf,
            FeatureWeights(),
        )
