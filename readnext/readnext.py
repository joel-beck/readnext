from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import InferenceData, InferenceDataConstructor, LanguageModelChoice
from readnext.utils import suppress_transformers_logging


def readnext(
    *,
    semanticscholar_id: str | None = None,
    semanticscholar_url: str | None = None,
    arxiv_id: str | None = None,
    arxiv_url: str | None = None,
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
