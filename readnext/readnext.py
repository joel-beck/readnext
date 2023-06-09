from pydantic import HttpUrl

from readnext import FeatureWeights, LanguageModelChoice
from readnext.inference import InferenceData, InferenceDataConstructor
from readnext.utils import suppress_transformers_logging


def readnext(
    *,
    semanticscholar_id: str | None = None,
    semanticscholar_url: HttpUrl | str | None = None,
    arxiv_id: str | None = None,
    arxiv_url: HttpUrl | str | None = None,
    language_model_choice: LanguageModelChoice,
    feature_weights: FeatureWeights,
) -> InferenceData:
    """
    # TODO: Add docstring
    """
    suppress_transformers_logging()

    constructor = InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=semanticscholar_url,
        arxiv_id=arxiv_id,
        arxiv_url=arxiv_url,
        language_model_choice=language_model_choice,
        feature_weights=feature_weights,
    )

    return InferenceData.from_constructor(constructor)
