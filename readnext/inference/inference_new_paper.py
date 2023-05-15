import os

import pandas as pd
from dotenv import load_dotenv

from readnext.config import DataPaths
from readnext.evaluation.metrics import (
    CosineSimilarity,
    CountCommonCitations,
    CountCommonReferences,
)
from readnext.evaluation.scoring import HybridScorer
from readnext.inference.inference_new_paper_base import (
    QueryCitationModelDataConstructor,
    QueryLanguageModelDataConstructor,
    send_semanticscholar_request,
)
from readnext.inference.inference_new_paper_language import select_query_embedding_function
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    DocumentScore,
    LanguageModelData,
)
from readnext.modeling.citation_models import (
    add_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling.language_models import LanguageModelChoice, load_embeddings_from_choice
from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
    load_df_from_pickle,
)

# BOOKMARK: Inputs
semanticscholar_url = (
    "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
)
arxiv_url = "https://arxiv.org/abs/1706.03762"
language_model_choice = LanguageModelChoice.word2vec


# SECTION: Collect Data
load_dotenv()

documents_authors_labels_citations_most_cited: pd.DataFrame = load_df_from_pickle(
    DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
).set_index("document_id")

SEMANTICSCHOLAR_API_KEY = os.getenv("SEMANTICSCHOLAR_API_KEY", "")
request_headers = {"x-api-key": SEMANTICSCHOLAR_API_KEY}

# with semanticscholar_url
response = send_semanticscholar_request(
    semanticscholar_id=get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url),
    request_headers=request_headers,
)

# with arxiv_url
response = send_semanticscholar_request(
    arxiv_id=get_arxiv_id_from_arxiv_url(arxiv_url), request_headers=request_headers
)

# SECTION: Citation Model
query_common_citations_urls = (
    [
        get_semanticscholar_url_from_semanticscholar_id(citation["paperId"])
        for citation in response.citations
    ]
    if response.citations is not None
    else []
)

query_common_references_urls = (
    [
        get_semanticscholar_url_from_semanticscholar_id(reference["paperId"])
        for reference in response.references
    ]
    if response.references is not None
    else []
)


num_common_citations: list[DocumentScore] = []

for document_id, candidate_citation_urls in documents_authors_labels_citations_most_cited[
    "citations"
].items():
    document_info = DocumentInfo(document_id=document_id)
    score = CountCommonCitations.count_common_values(
        query_common_citations_urls, candidate_citation_urls
    )

    document_score = DocumentScore(document_info=document_info, score=score)
    num_common_citations.append(document_score)


query_co_citation_analysis_scores = pd.DataFrame(
    {"score": [num_common_citations]}, index=[-1]
).rename_axis("document_id", axis="index")


# the same for common references
num_common_references: list[DocumentScore] = []

for document_id, candidate_reference_urls in documents_authors_labels_citations_most_cited[
    "references"
].items():
    document_info = DocumentInfo(document_id=document_id)
    score = CountCommonReferences.count_common_values(
        query_common_references_urls, candidate_reference_urls
    )

    document_score = DocumentScore(document_info=document_info, score=score)
    num_common_references.append(document_score)

query_bibliographic_coupling_scores = pd.DataFrame(
    {"score": [num_common_references]}, index=[-1]
).rename_axis("document_id", axis="index")


query_citation_model_data_constructor = QueryCitationModelDataConstructor(
    response=response,
    query_document_id=-1,
    documents_data=documents_authors_labels_citations_most_cited.pipe(add_feature_rank_cols).pipe(
        set_missing_publication_dates_to_max_rank
    ),
    co_citation_analysis_scores=query_co_citation_analysis_scores,
    bibliographic_coupling_scores=query_bibliographic_coupling_scores,
)
query_citation_model_data = CitationModelData.from_constructor(
    query_citation_model_data_constructor
)


# SECTION: Language Model
assert response.abstract is not None

# set document id of unseen document to -1
query_document_info = DocumentInfo(document_id=-1, abstract=response.abstract)

query_embedding_function = select_query_embedding_function(language_model_choice)

query_abstract_embedding = query_embedding_function(query_document_info)

candidate_embeddings: pd.DataFrame = load_embeddings_from_choice(language_model_choice)

cosine_similarity_scores: list[DocumentScore] = []

for document_id, candidate_embedding in zip(
    candidate_embeddings["document_id"], candidate_embeddings["embedding"]
):
    document_info = DocumentInfo(document_id=document_id)
    score = CosineSimilarity.score(query_abstract_embedding, candidate_embedding)

    document_score = DocumentScore(document_info=document_info, score=score)
    cosine_similarity_scores.append(document_score)

query_cosine_similarities = pd.DataFrame(
    {"score": [cosine_similarity_scores]}, index=[-1]
).rename_axis("document_id", axis="index")

query_language_model_data_constructor = QueryLanguageModelDataConstructor(
    response=response,
    query_document_id=-1,
    documents_data=documents_authors_labels_citations_most_cited,
    cosine_similarities=query_cosine_similarities,
)
query_language_model_data = LanguageModelData.from_constructor(
    query_language_model_data_constructor
)

# TODO: Does not work right now!
query_hybrid_scorer = HybridScorer(
    language_model_name=language_model_choice.name,
    language_model_data=query_language_model_data,
    citation_model_data=query_citation_model_data,
)
query_hybrid_scorer.recommend()

query_hybrid_scorer.citation_to_language_candidates
query_hybrid_scorer.citation_to_language_recommendations
query_hybrid_scorer.language_to_citation_candidates
query_hybrid_scorer.language_to_citation_recommendations
