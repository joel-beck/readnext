from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import InferenceData, InferenceDataConstructor, LanguageModelChoice
from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
    suppress_transformers_logging,
)


def main() -> None:
    suppress_transformers_logging()

    # SECTION: Seen Paper
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )
    semanticscholar_id = get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url)
    get_semanticscholar_url_from_semanticscholar_id(semanticscholar_id)
    arxiv_url = "https://arxiv.org/abs/1706.03762"
    arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

    # SUBSECTION: Input is semanticscholar ID
    inference_data_constructor_seen_semanticscholar_id = InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_seen_from_semanticscholar_id = InferenceData.from_constructor(
        inference_data_constructor_seen_semanticscholar_id
    )

    print(inference_data_seen_from_semanticscholar_id.document_identifier)
    print(inference_data_seen_from_semanticscholar_id.document_info)
    print(inference_data_seen_from_semanticscholar_id.features)
    print(inference_data_seen_from_semanticscholar_id.ranks)
    print(inference_data_seen_from_semanticscholar_id.points)
    print(inference_data_seen_from_semanticscholar_id.labels)
    print(
        inference_data_seen_from_semanticscholar_id.recommendations.citation_to_language_candidates
    )
    print(
        inference_data_seen_from_semanticscholar_id.recommendations.language_to_citation_candidates
    )

    # SUBSECTION: Input is semanticscholar URL
    inference_data_constructor_seen_semanticscholar_url = InferenceDataConstructor(
        semanticscholar_url=semanticscholar_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_seen_from_semanticscholar_url = InferenceData.from_constructor(
        inference_data_constructor_seen_semanticscholar_url
    )

    print(inference_data_seen_from_semanticscholar_url.document_identifier)
    print(inference_data_seen_from_semanticscholar_url.document_info)
    print(inference_data_seen_from_semanticscholar_url.features)
    print(inference_data_seen_from_semanticscholar_url.ranks)
    print(inference_data_seen_from_semanticscholar_url.points)
    print(inference_data_seen_from_semanticscholar_url.labels)
    print(inference_data_seen_from_semanticscholar_url.recommendations.citation_to_language)

    # SUBSECTION: Input is arxiv ID
    inference_data_constructor_seen_arxiv_id = InferenceDataConstructor(
        arxiv_id=arxiv_id,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_seen_from_arxiv_id = InferenceData.from_constructor(
        inference_data_constructor_seen_arxiv_id
    )

    print(inference_data_seen_from_arxiv_id.document_identifier)
    print(inference_data_seen_from_arxiv_id.document_info)
    print(inference_data_seen_from_arxiv_id.features)
    print(inference_data_seen_from_arxiv_id.ranks)
    print(inference_data_seen_from_arxiv_id.points)
    print(inference_data_seen_from_arxiv_id.labels)
    print(inference_data_seen_from_arxiv_id.recommendations.citation_to_language)

    # SUBSECTION: Input is arxiv URL
    inference_data_constructor_seen_arxiv_url = InferenceDataConstructor(
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
    )
    inference_data_seen_from_arxiv_url = InferenceData.from_constructor(
        inference_data_constructor_seen_arxiv_url
    )

    print(inference_data_seen_from_arxiv_url.document_identifier)
    print(inference_data_seen_from_arxiv_url.document_info)
    print(inference_data_seen_from_arxiv_url.features)
    print(inference_data_seen_from_arxiv_url.ranks)
    print(inference_data_seen_from_arxiv_url.points)
    print(inference_data_seen_from_arxiv_url.labels)
    print(inference_data_seen_from_arxiv_url.recommendations.citation_to_language)

    # SECTION: Unseen Paper
    arxiv_url = "https://arxiv.org/abs/2303.08774"
    arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

    inference_data_constructor_unseen_arxiv_id = InferenceDataConstructor(
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.longformer,
        feature_weights=FeatureWeights(),
    )

    inference_data_unseen_from_arxiv_id = InferenceData.from_constructor(
        inference_data_constructor_unseen_arxiv_id
    )

    print(inference_data_unseen_from_arxiv_id.document_identifier)
    print(inference_data_unseen_from_arxiv_id.document_info)
    print(inference_data_unseen_from_arxiv_id.features)
    print(inference_data_unseen_from_arxiv_id.ranks)
    print(inference_data_unseen_from_arxiv_id.points)
    print(inference_data_unseen_from_arxiv_id.labels)
    print(inference_data_unseen_from_arxiv_id.recommendations.citation_to_language)
    print(inference_data_unseen_from_arxiv_id.recommendations.citation_to_language_candidates)
    print(inference_data_unseen_from_arxiv_id.recommendations.language_to_citation)
    print(inference_data_unseen_from_arxiv_id.recommendations.language_to_citation_candidates)


if __name__ == "__main__":
    main()
