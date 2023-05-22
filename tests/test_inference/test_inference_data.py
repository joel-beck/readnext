import numpy as np
import pandas as pd
from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import (
    InferenceData,
    InferenceDataConstructor,
    LanguageModelChoice,
    DocumentIdentifier,
)

# These imports must not come from `readnext.inference`, otherwise they are really
# imported twice with different module scopes and `isinstance()` checks fail.
from readnext.inference.inference_data_constructor import Features, Labels, Ranks, Recommendations
from readnext.modeling import DocumentInfo
from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
    suppress_transformers_logging,
)


# SECTION: Seen Paper
suppress_transformers_logging()

semanticscholar_url = (
    "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
)
semanticscholar_id = get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url)
get_semanticscholar_url_from_semanticscholar_id(semanticscholar_id)
arxiv_url = "https://arxiv.org/abs/1706.03762"
arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

inference_data_seen_constructor_semanticscholar_id = InferenceDataConstructor(
    semanticscholar_id=semanticscholar_id,
    language_model_choice=LanguageModelChoice.tfidf,
    feature_weights=FeatureWeights(),
)
inference_data_seen_from_semanticscholar_id = InferenceData.from_constructor(
    inference_data_seen_constructor_semanticscholar_id
)


# SUBSECTION: Test Document Identifier
document_identifier = inference_data_seen_from_semanticscholar_id.document_identifier
assert isinstance(document_identifier, DocumentIdentifier)

assert isinstance(document_identifier.semanticscholar_url, str)
assert (
    document_identifier.semanticscholar_url
    == "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
)

assert isinstance(document_identifier.semanticscholar_id, str)
assert document_identifier.semanticscholar_id == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

assert isinstance(document_identifier.arxiv_url, str)
assert document_identifier.arxiv_url == "https://arxiv.org/abs/1706.03762"

assert isinstance(document_identifier.arxiv_id, str)
assert document_identifier.arxiv_id == "1706.03762"

assert isinstance(document_identifier.d3_document_id, int)
assert document_identifier.d3_document_id == 13756489

# SUBSECTION: Test Document Info
document_info = inference_data_seen_from_semanticscholar_id.document_info
assert isinstance(document_info, DocumentInfo)

assert isinstance(document_info.d3_document_id, int)
assert document_info.d3_document_id == 13756489

assert isinstance(document_info.title, str)
assert document_info.title == "Attention is All you Need"

assert isinstance(document_info.author, str)
assert document_info.author == "Lukasz Kaiser"

assert isinstance(document_info.abstract, str)
# abstract is not set for seen documents
# TODO: Should this be the case?
assert len(document_info.abstract) == 0

assert isinstance(document_info.arxiv_labels, list)
assert all(isinstance(label, str) for label in document_info.arxiv_labels)
assert document_info.arxiv_labels == ["cs.CL", "cs.LG"]


# SUBSECTION: Test Features
features = inference_data_seen_from_semanticscholar_id.features
assert isinstance(features, Features)

assert isinstance(features.publication_date, pd.Series)
assert features.publication_date.name == "publication_date"
assert features.publication_date.dtype == pd.StringDtype()
assert features.publication_date.index.name == "document_id"
assert features.publication_date.index.dtype == pd.Int64Dtype()

assert isinstance(features.citationcount_document, pd.Series)
assert features.citationcount_document.name == "citationcount_document"
assert features.citationcount_document.dtype == pd.Int64Dtype()
assert features.citationcount_document.index.name == "document_id"
assert features.citationcount_document.index.dtype == pd.Int64Dtype()

assert isinstance(features.citationcount_author, pd.Series)
assert features.citationcount_author.name == "citationcount_author"
assert features.citationcount_author.dtype == pd.Int64Dtype()
assert features.citationcount_author.index.name == "document_id"
assert features.citationcount_author.index.dtype == pd.Int64Dtype()

assert isinstance(features.co_citation_analysis, pd.Series)
assert features.co_citation_analysis.name == "co_citation_analysis"
assert features.co_citation_analysis.dtype == np.dtype("int64")
assert features.co_citation_analysis.index.name == "document_id"
assert features.co_citation_analysis.index.dtype == pd.Int64Dtype()

assert isinstance(features.bibliographic_coupling, pd.Series)
assert features.bibliographic_coupling.name == "bibliographic_coupling"
assert features.bibliographic_coupling.dtype == np.dtype("int64")
assert features.bibliographic_coupling.index.name == "document_id"
assert features.bibliographic_coupling.index.dtype == pd.Int64Dtype()

assert isinstance(features.cosine_similarity, pd.Series)
assert features.cosine_similarity.name == "cosine_similarity"
assert features.cosine_similarity.dtype == np.dtype("float64")
assert features.cosine_similarity.index.name == "document_id"
assert features.cosine_similarity.index.dtype == pd.Int64Dtype()

assert isinstance(features.feature_weights, FeatureWeights)


# SUBSECTION: Test Ranks
ranks = inference_data_seen_from_semanticscholar_id.ranks
assert isinstance(ranks, Ranks)

assert isinstance(ranks.publication_date, pd.Series)
assert ranks.publication_date.name == "publication_date_rank"
assert ranks.publication_date.dtype == np.dtype("float64")
assert ranks.publication_date.index.name == "document_id"
assert ranks.publication_date.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.citationcount_document, pd.Series)
assert ranks.citationcount_document.name == "citationcount_document_rank"
assert ranks.citationcount_document.dtype == np.dtype("float64")
assert ranks.citationcount_document.index.name == "document_id"
assert ranks.citationcount_document.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.citationcount_author, pd.Series)
assert ranks.citationcount_author.name == "citationcount_author_rank"
assert ranks.citationcount_author.dtype == np.dtype("float64")
assert ranks.citationcount_author.index.name == "document_id"
assert ranks.citationcount_author.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.co_citation_analysis, pd.Series)
assert ranks.co_citation_analysis.name == "co_citation_analysis_rank"
assert ranks.co_citation_analysis.dtype == np.dtype("float64")
assert ranks.co_citation_analysis.index.name == "document_id"
assert ranks.co_citation_analysis.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.bibliographic_coupling, pd.Series)
assert ranks.bibliographic_coupling.name == "bibliographic_coupling_rank"
assert ranks.bibliographic_coupling.dtype == np.dtype("float64")
assert ranks.bibliographic_coupling.index.name == "document_id"
assert ranks.bibliographic_coupling.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.cosine_similarity, pd.Series)
assert ranks.cosine_similarity.name == "cosine_similarity_rank"
assert ranks.cosine_similarity.dtype == np.dtype("float64")
assert ranks.cosine_similarity.index.name == "document_id"
assert ranks.cosine_similarity.index.dtype == np.dtype("int64")

# SUBSECTION: Test Labels
labels = inference_data_seen_from_semanticscholar_id.labels
assert isinstance(labels, Labels)

assert isinstance(labels.arxiv, pd.Series)
assert labels.arxiv.name == "arxiv_labels"
assert labels.arxiv.dtype == object
assert labels.arxiv.index.name == "document_id"
assert labels.arxiv.index.dtype == pd.Int64Dtype()
assert all(isinstance(labels, list) for labels in labels.arxiv)
assert all(isinstance(label, str) for labels in labels.arxiv for label in labels)
assert all(len(labels) > 0 for labels in labels.arxiv)

assert isinstance(labels.integer, pd.Series)
assert labels.integer.name == "integer_labels"
assert labels.integer.dtype == np.dtype("int64")
assert labels.integer.index.name == "document_id"
assert labels.integer.index.dtype == pd.Int64Dtype()
assert labels.integer.unique().tolist() == [0, 1]

# SUBSECTION: Test Recommendations
recommendations = inference_data_seen_from_semanticscholar_id.recommendations
assert isinstance(recommendations, Recommendations)
# NOTE: For all of the four data frames the same tests

assert isinstance(recommendations.citation_to_language, pd.DataFrame)
assert isinstance(recommendations.citation_to_language_candidates, pd.DataFrame)
assert isinstance(recommendations.language_to_citation, pd.DataFrame)
assert isinstance(recommendations.language_to_citation_candidates, pd.DataFrame)

assert recommendations.citation_to_language.index.name == "document_id"
assert recommendations.citation_to_language.index.dtype == pd.Int64Dtype()
assert recommendations.citation_to_language.shape[1] == 4
assert recommendations.citation_to_language.columns.tolist() == [
    "cosine_similarity",
    "title",
    "author",
    "arxiv_labels",
]
assert recommendations.citation_to_language.dtypes.tolist() == [
    np.dtype("float64"),
    pd.StringDtype(),
    pd.StringDtype(),
    object,
]

# check that cosine similarities are between 0 and 1
assert recommendations.citation_to_language.cosine_similarity.min() >= 0
assert recommendations.citation_to_language.cosine_similarity.max() <= 1

# check that the cosine similarities are sorted in descending order
assert all(recommendations.citation_to_language.cosine_similarity.diff().dropna().values <= 0)  # type: ignore


# SECTION: Unseen Paper
arxiv_url = "https://arxiv.org/abs/2303.08774"
arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

inference_data_unseen_constructor_arxiv_id = InferenceDataConstructor(
    arxiv_url=arxiv_url,
    language_model_choice=LanguageModelChoice.scibert,
    feature_weights=FeatureWeights(),
)
inference_data_unseen_from_arxiv_id = InferenceData.from_constructor(
    inference_data_unseen_constructor_arxiv_id
)

# SUBSECTION: Test Document Identifier
document_identifier = inference_data_unseen_from_arxiv_id.document_identifier
assert isinstance(document_identifier, DocumentIdentifier)

assert isinstance(document_identifier.semanticscholar_url, str)
assert (
    document_identifier.semanticscholar_url
    == "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
)

assert isinstance(document_identifier.semanticscholar_id, str)
assert document_identifier.semanticscholar_id == "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

assert isinstance(document_identifier.arxiv_url, str)
assert document_identifier.arxiv_url == "https://arxiv.org/abs/2303.08774"

assert isinstance(document_identifier.arxiv_id, str)
assert document_identifier.arxiv_id == "2303.08774"

assert isinstance(document_identifier.d3_document_id, int)
assert document_identifier.d3_document_id == -1

# SUBSECTION: Test Document Info
document_info = inference_data_unseen_from_arxiv_id.document_info
assert isinstance(document_info, DocumentInfo)

assert isinstance(document_info.d3_document_id, int)
assert document_info.d3_document_id == -1

assert isinstance(document_info.title, str)
assert document_info.title == "GPT-4 Technical Report"

assert isinstance(document_info.author, str)
assert document_info.author == ""

assert isinstance(document_info.abstract, str)
assert len(document_info.abstract) > 0

assert isinstance(document_info.arxiv_labels, list)
assert document_info.arxiv_labels == []


# SUBSECTION: Test Features
features = inference_data_unseen_from_arxiv_id.features
assert isinstance(features, Features)

assert isinstance(features.publication_date, pd.Series)
assert features.publication_date.name == "publication_date"
assert features.publication_date.dtype == pd.StringDtype()
assert features.publication_date.index.name == "document_id"
assert features.publication_date.index.dtype == pd.Int64Dtype()

assert isinstance(features.citationcount_document, pd.Series)
assert features.citationcount_document.name == "citationcount_document"
assert features.citationcount_document.dtype == pd.Int64Dtype()
assert features.citationcount_document.index.name == "document_id"
assert features.citationcount_document.index.dtype == pd.Int64Dtype()

assert isinstance(features.citationcount_author, pd.Series)
assert features.citationcount_author.name == "citationcount_author"
assert features.citationcount_author.dtype == pd.Int64Dtype()
assert features.citationcount_author.index.name == "document_id"
assert features.citationcount_author.index.dtype == pd.Int64Dtype()

assert isinstance(features.co_citation_analysis, pd.Series)
assert features.co_citation_analysis.name == "co_citation_analysis"
assert features.co_citation_analysis.dtype == np.dtype("int64")
assert features.co_citation_analysis.index.name == "document_id"
assert features.co_citation_analysis.index.dtype == pd.Int64Dtype()

assert isinstance(features.bibliographic_coupling, pd.Series)
assert features.bibliographic_coupling.name == "bibliographic_coupling"
assert features.bibliographic_coupling.dtype == np.dtype("int64")
assert features.bibliographic_coupling.index.name == "document_id"
assert features.bibliographic_coupling.index.dtype == pd.Int64Dtype()

assert isinstance(features.cosine_similarity, pd.Series)
assert features.cosine_similarity.name == "cosine_similarity"
assert features.cosine_similarity.dtype == np.dtype("float64")
assert features.cosine_similarity.index.name == "document_id"
assert features.cosine_similarity.index.dtype == pd.Int64Dtype()

assert isinstance(features.feature_weights, FeatureWeights)

# SUBSECTION: Test Ranks
ranks = inference_data_unseen_from_arxiv_id.ranks
assert isinstance(ranks, Ranks)

assert isinstance(ranks.publication_date, pd.Series)
assert ranks.publication_date.name == "publication_date_rank"
assert ranks.publication_date.dtype == np.dtype("float64")
assert ranks.publication_date.index.name == "document_id"
assert ranks.publication_date.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.citationcount_document, pd.Series)
assert ranks.citationcount_document.name == "citationcount_document_rank"
assert ranks.citationcount_document.dtype == np.dtype("float64")
assert ranks.citationcount_document.index.name == "document_id"
assert ranks.citationcount_document.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.citationcount_author, pd.Series)
assert ranks.citationcount_author.name == "citationcount_author_rank"
assert ranks.citationcount_author.dtype == np.dtype("float64")
assert ranks.citationcount_author.index.name == "document_id"
assert ranks.citationcount_author.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.co_citation_analysis, pd.Series)
assert ranks.co_citation_analysis.name == "co_citation_analysis_rank"
assert ranks.co_citation_analysis.dtype == np.dtype("float64")
assert ranks.co_citation_analysis.index.name == "document_id"
assert ranks.co_citation_analysis.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.bibliographic_coupling, pd.Series)
assert ranks.bibliographic_coupling.name == "bibliographic_coupling_rank"
assert ranks.bibliographic_coupling.dtype == np.dtype("float64")
assert ranks.bibliographic_coupling.index.name == "document_id"
assert ranks.bibliographic_coupling.index.dtype == pd.Int64Dtype()

assert isinstance(ranks.cosine_similarity, pd.Series)
assert ranks.cosine_similarity.name == "cosine_similarity_rank"
assert ranks.cosine_similarity.dtype == np.dtype("float64")
assert ranks.cosine_similarity.index.name == "document_id"
assert ranks.cosine_similarity.index.dtype == np.dtype("int64")

# SUBSECTION: Test Labels
labels = inference_data_unseen_from_arxiv_id.labels
assert isinstance(labels, Labels)

assert isinstance(labels.arxiv, pd.Series)
assert labels.arxiv.name == "arxiv_labels"
assert labels.arxiv.dtype == object
assert labels.arxiv.index.name == "document_id"
assert labels.arxiv.index.dtype == pd.Int64Dtype()
assert all(isinstance(labels, list) for labels in labels.arxiv)
assert all(isinstance(label, str) for labels in labels.arxiv for label in labels)
assert all(len(labels) > 0 for labels in labels.arxiv)

assert isinstance(labels.integer, pd.Series)
assert labels.integer.name == "integer_labels"
assert labels.integer.dtype == np.dtype("int64")
assert labels.integer.index.name == "document_id"
assert labels.integer.index.dtype == pd.Int64Dtype()
# for unseen documents no arxiv label is present, thus the label cannot coincide with
# any of the candidate labels
assert labels.integer.unique().tolist() == [0]

# SUBSECTION: Test Recommendations
recommendations = inference_data_unseen_from_arxiv_id.recommendations
assert isinstance(recommendations, Recommendations)
# NOTE: For all of the four data frames the same tests

assert isinstance(recommendations.citation_to_language, pd.DataFrame)
assert isinstance(recommendations.citation_to_language_candidates, pd.DataFrame)
assert isinstance(recommendations.language_to_citation, pd.DataFrame)
assert isinstance(recommendations.language_to_citation_candidates, pd.DataFrame)

assert recommendations.citation_to_language.index.name == "document_id"
assert recommendations.citation_to_language.index.dtype == pd.Int64Dtype()
assert recommendations.citation_to_language.shape[1] == 4
assert recommendations.citation_to_language.columns.tolist() == [
    "cosine_similarity",
    "title",
    "author",
    "arxiv_labels",
]
assert recommendations.citation_to_language.dtypes.tolist() == [
    np.dtype("float64"),
    pd.StringDtype(),
    pd.StringDtype(),
    object,
]

# check that cosine similarities are between 0 and 1
assert recommendations.citation_to_language.cosine_similarity.min() >= 0
assert recommendations.citation_to_language.cosine_similarity.max() <= 1

# check that the cosine similarities are sorted in descending order
assert all(recommendations.citation_to_language.cosine_similarity.diff().dropna().values <= 0)  # type: ignore
