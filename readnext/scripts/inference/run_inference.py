from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import InferenceData, InferenceDataConstructor, LanguageModelChoice
from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_id_from_semanticscholar_url,
)


def main() -> None:
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    semanticscholar_id = get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url)
    arxiv_url = "https://arxiv.org/abs/1706.03762"
    arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

    # SUBSECTION: Input is semanticscholar ID
    inference_data_constructor_semanticscholar_id = InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_query_id = InferenceData.from_constructor(
        inference_data_constructor_semanticscholar_id
    )
    print(inference_data_query_id.recommendations.citation_to_language_candidates)

    # SUBSECTION: Input is semanticscholar URL
    inference_data_constructor_semanticscholar_url = InferenceDataConstructor(
        semanticscholar_url=semanticscholar_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_semanticscholar_url = InferenceData.from_constructor(
        inference_data_constructor_semanticscholar_url
    )
    print(inference_data_semanticscholar_url.recommendations.citation_to_language_candidates)

    # SUBSECTION: Input is arxiv ID
    inference_data_constructor_arxiv_id = InferenceDataConstructor(
        arxiv_id=arxiv_id,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_arxiv_url = InferenceData.from_constructor(inference_data_constructor_arxiv_id)
    print(inference_data_arxiv_url.recommendations.citation_to_language_candidates)

    # SUBSECTION: Input is arxiv URL
    inference_data_constructor_arxiv_url = InferenceDataConstructor(
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_arxiv_url = InferenceData.from_constructor(inference_data_constructor_arxiv_url)
    print(inference_data_arxiv_url.recommendations.citation_to_language_candidates)


if __name__ == "__main__":
    main()
