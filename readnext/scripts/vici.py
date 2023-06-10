from readnext import readnext, LanguageModelChoice, FeatureWeights

# result = readnext(
#     arxiv_url="https://arxiv.org/abs/1706.03762",
#     language_model_choice=LanguageModelChoice.GLOVE,
#     feature_weights=FeatureWeights(citationcount_document=8, publication_date=0),
# )

# result.recommendations.citation_to_language.select("title")[0].item()


result = readnext(
    arxiv_url="https://arxiv.org/abs/2303.12712",
    language_model_choice=LanguageModelChoice.GLOVE,
    feature_weights=FeatureWeights(citationcount_document=8, publication_date=0),
)
