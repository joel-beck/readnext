import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import InferenceDataConstructor
from readnext.modeling.language_models import LanguageModelChoice


def test_kw_only_initialization_inference_data_constructor() -> None:
    with pytest.raises(TypeError):
        InferenceDataConstructor(
            -1,  # type: ignore
            "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "1905.12616",
            "https://arxiv.org/abs/1905.12616",
            LanguageModelChoice.tfidf,
            FeatureWeights(),
        )
