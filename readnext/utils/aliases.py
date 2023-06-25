from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import polars as pl
from numpy.typing import NDArray

Tokens: TypeAlias = list[str]
TokenIds: TypeAlias = list[int]
Embedding: TypeAlias = list[float]


# tokens_frame.columns = ["d3_document_id", "tokens"]
# data types: d3_document_id: int, tokens: list[str]
TokensFrame: TypeAlias = pl.DataFrame

# token_ids_frame.columns = ["d3_document_id", "token_ids"]
# data types: d3_document_id: int, token_ids: list[int]
TokenIdsFrame: TypeAlias = pl.DataFrame

# embeddings_frame.columns = ["d3_document_id", "embedding"]
# data types: d3_document_id: int, embedding: list[float]
EmbeddingsFrame: TypeAlias = pl.DataFrame

# scores_frame.columns = ["query_d3_document_id", "candidate_d3_document_id", "score"]
# data types: query_d3_document_id: int, candidate_d3_document_id: int, score: float
ScoresFrame = pl.DataFrame

# candidate_scores_frame.columns = ["candidate_d3_document_id", "score"]
# data types: candidate_d3_document_id: int, score: float
CandidateScoresFrame: TypeAlias = pl.DataFrame

# candidate_ranks_frame.columns = ["candidate_d3_document_id", "rank"]
# data types: candidate_d3_document_id: int, rank: int
CandidateRanksFrame: TypeAlias = pl.DataFrame

# contains all columns of the full documents dataset
DocumentsFrame: TypeAlias = pl.DataFrame

# contains all columns of the full documents dataset with the column
# `candidate_d3_document_id` instead of `d3_document_id`
QueryDocumentsFrame: TypeAlias = pl.DataFrame

# integer_labels_frame.columns = ["candidate_d3_document_id", "integer_label"]
# data types: candidate_d3_document_id: int, integer_label: int
IntegerLabelsFrame: TypeAlias = pl.DataFrame

# citation_info_frame.columns = ["candidate_d3_document_id", "title", "author", "arxiv_labels",
# "semanticscholar_url", "arxiv_url"]
# data types: candidate_d3_document_id: int, title: str, author: str, arxiv_labels:
# list[str], semanticscholar_url: str, arxiv_url: str
CitationInfoFrame: TypeAlias = pl.DataFrame

# language_info_frame.columns = ["candidate_d3_document_id", "title", "author",
# "publication_date", "arxiv_labels", "semanticscholar_url", "arxiv_url"]
# data types: candidate_d3_document_id: int, title: str, author: str, publication_date:
# str, arxiv_labels: list[str], semanticscholar_url: str, arxiv_url: str
LanguageInfoFrame: TypeAlias = pl.DataFrame

# citation_features_frame.columns = ["candidate_d3_document_id","publication_date",
# "citationcount_document", "citationcount_author", "co_citation_analysis_score",
# "bibliographic_coupling_score"]
# data types: candidate_d3_document_id: int, publication_date: str,
# citationcount_document: int, citationcount_author: int, co_citation_analysis_score:
# int, bibliographic_coupling_score: int
CitationFeaturesFrame: TypeAlias = pl.DataFrame

# language_features_frame.columns = ["candidate_d3_document_id", "cosine_similarity"]
# data types: candidate_d3_document_id: int, cosine_similarity: float
LanguageFeaturesFrame: TypeAlias = pl.DataFrame

# citation_ranks_frame.columns = ["candidate_d3_document_id", "publication_date_rank",
# "citationcount_document_rank", "citationcount_author_rank",
# "co_citation_analysis_rank", "bibliographic_coupling_rank"]
# data types: candidate_d3_document_id: int, publication_date_rank: int,
# citationcount_document_rank: int, citationcount_author_rank: int,
# co_citation_analysis_rank: int, bibliographic_coupling_rank: int
CitationRanksFrame: TypeAlias = pl.DataFrame

# language_ranks_frame.columns = ["candidate_d3_document_id",
# "publication_date_points", "citationcount_document_points",
# "citationcount_author_points", "co_citation_analysis_points",
# "bibliographic_coupling_points"]
# data types: candidate_d3_document_id: int, publication_date_points: int,
# citationcount_document_points: int, citationcount_author_points: int,
# co_citation_analysis_points: int, bibliographic_coupling_points: int
CitationPointsFrame: TypeAlias = pl.DataFrame

# tfidf and bm25
KeywordAlgorithm: TypeAlias = Callable[[Tokens, Sequence[Tokens]], np.ndarray]

# language model embedding functions during inference
QueryEmbeddingFunction: TypeAlias = Callable[[pl.DataFrame], Embedding]

Vector: TypeAlias = Sequence | NDArray | pl.Series
EmbeddingVector: TypeAlias = Sequence[float] | NDArray | pl.Series

IntegerLabelList: TypeAlias = Sequence[int] | NDArray | pl.Series
IntegerLabelLists: TypeAlias = Sequence[IntegerLabelList]

StringLabelList: TypeAlias = Sequence[str] | pl.Series
StringLabelLists: TypeAlias = Sequence[StringLabelList] | pl.Series
