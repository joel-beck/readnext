from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import InferenceData, InferenceDataConstructor, LanguageModelChoice


def main() -> None:
    query_id = 13756489
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    arxiv_url = "https://arxiv.org/abs/1706.03762"
    paper_title = "Attention is All you Need"

    # SUBSECTION: Input is query ID
    inference_data_constructor_query_id = InferenceDataConstructor(
        query_id=query_id,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_query_id = InferenceData.from_constructor(inference_data_constructor_query_id)
    inference_data_query_id.recommendations.citation_to_language_candidates

    # SUBSECTION: Input is semanticscholar URL
    inference_data_constructor_semanticscholar_url = InferenceDataConstructor(
        semanticscholar_url=semanticscholar_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_semanticscholar_url = InferenceData.from_constructor(
        inference_data_constructor_semanticscholar_url
    )
    inference_data_semanticscholar_url.recommendations.citation_to_language_candidates

    # SUBSECTION: Input is arxiv URL
    inference_data_constructor_arxiv_url = InferenceDataConstructor(
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_arxiv_url = InferenceData.from_constructor(inference_data_constructor_arxiv_url)
    inference_data_arxiv_url.recommendations.citation_to_language_candidates

    # SUBSECTION: Input is paper title
    inference_data_constructor_paper_title = InferenceDataConstructor(
        paper_title=paper_title,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_paper_title = InferenceData.from_constructor(
        inference_data_constructor_paper_title
    )
    inference_data_paper_title.recommendations.citation_to_language_candidates


if __name__ == "__main__":
    main()
