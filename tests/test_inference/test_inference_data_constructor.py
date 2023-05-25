import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, InferenceDataConstructor
from readnext.inference.attribute_getter import SeenPaperAttributeGetter, UnseenPaperAttributeGetter
from readnext.inference.inference_data_constructor import Features, Labels, Ranks, Recommendations
from readnext.modeling.language_models import LanguageModelChoice

seen_inference_data_constructors = ["inference_data_seen_constructor_from_semanticscholar_id"]
unseen_inference_data_constructors = ["inference_data_unseen_constructor_from_arxiv_url"]
inference_data_constructors = seen_inference_data_constructors + unseen_inference_data_constructors


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "inference_data_constructor", lazy_fixture(seen_inference_data_constructors)
)
def test_seen_attribute_getter_is_selected_correctly(
    inference_data_constructor: InferenceDataConstructor,
) -> None:
    assert inference_data_constructor.query_document_in_training_data()
    assert isinstance(
        inference_data_constructor.attribute_getter,
        SeenPaperAttributeGetter,
    )


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "inference_data_constructor", lazy_fixture(unseen_inference_data_constructors)
)
def test_unseen_attribute_getter_is_selected_correctly(
    inference_data_constructor: InferenceDataConstructor,
) -> None:
    assert not inference_data_constructor.query_document_in_training_data()
    assert isinstance(
        inference_data_constructor.attribute_getter,
        UnseenPaperAttributeGetter,
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
            pd.Series(),  # type: ignore
            pd.Series(),
            pd.Series(),
            pd.Series(),
            pd.Series(),
            pd.Series(),
            FeatureWeights(),
        )


def test_kw_only_initialization_ranks() -> None:
    with pytest.raises(TypeError):
        Ranks(
            pd.Series(),  # type: ignore
            pd.Series(),
            pd.Series(),
            pd.Series(),
            pd.Series(),
            pd.Series(),
        )


def test_kw_only_initialization_labels() -> None:
    with pytest.raises(TypeError):
        Labels(pd.Series(), pd.Series())  # type: ignore


def test_kw_only_initialization_recommendations() -> None:
    with pytest.raises(TypeError):
        Recommendations(
            pd.DataFrame(),  # type: ignore
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
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
