import pandas as pd
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, InferenceDataConstructor
from readnext.inference.inference_data_constructor import Features, Labels, Ranks, Recommendations
from readnext.modeling.language_models import LanguageModelChoice


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
