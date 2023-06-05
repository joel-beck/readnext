from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import polars as pl
from numpy.typing import NDArray

Tokens: TypeAlias = list[str]
TokenIds: TypeAlias = list[int]
Embedding: TypeAlias = list[float]

# tokens_frame.columns = ["d3_document_id", "tokens"]
TokensFrame: TypeAlias = pl.DataFrame
# token_ids_frame.columns = ["d3_document_id", "token_ids"]
TokenIdsFrame: TypeAlias = pl.DataFrame
# embeddings_frame.columns = ["d3_document_id", "embedding"]
EmbeddingsFrame: TypeAlias = pl.DataFrame
# scores_frame.columns = ["query_d3_document_id", "candidate_d3_document_id", "score"]
ScoresFrame = pl.DataFrame
# candidate_scores_frame.columns = ["candidate_d3_document_id", "score"]
CandidateScoresFrame: TypeAlias = pl.DataFrame
# candidate_ranks_frame.columns = ["d3_document_id", "rank"]
CandidateRanksFrame: TypeAlias = pl.DataFrame

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
