from readnext import readnext, LanguageModelChoice, FeatureWeights

result = readnext(
    arxiv_id="2101.03041",
    language_model_choice=LanguageModelChoice.BM25,
    feature_weights=FeatureWeights(publication_date=-1),
)


# semanticscholar_url = result.recommendations.citation_to_language[0, "semanticscholar_url"]

# next_result = readnext(
#     semanticscholar_url=semanticscholar_url,
#     language_model_choice=LanguageModelChoice.SCIBERT,
#     feature_weights=FeatureWeights(
#         citationcount_author=0, co_citation_analysis=3, bibliographic_coupling=3
#     ),
# )

# print(next_result.recommendations.language_to_citation.to_pandas().to_markdown(index=False))
