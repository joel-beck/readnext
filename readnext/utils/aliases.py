from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import polars as pl
from numpy.typing import NDArray

from readnext.modeling.document_info import DocumentInfo

Tokens: TypeAlias = list[str]
TokenIds: TypeAlias = list[int]
Embedding: TypeAlias = list[float]

# data frame with two columns named `d3_document_id` and `tokens`
TokensFrame: TypeAlias = pl.DataFrame
# data frame with two columns named `d3_document_id` and `token_ids`
TokenIdsFrame: TypeAlias = pl.DataFrame
# data frame with two columns named `d3_document_id` and `embedding`
EmbeddingsFrame: TypeAlias = pl.DataFrame
# data frame with a three columns named `query_d3_document_id`,
# `candidate_d3_document_id` and `score`.
ScoresFrame = pl.DataFrame

# tfidf and bm25
KeywordAlgorithm: TypeAlias = Callable[[Tokens, Sequence[Tokens]], np.ndarray]

# language model embedding functions during inference
QueryEmbeddingFunction: TypeAlias = Callable[[DocumentInfo], Embedding]

Vector: TypeAlias = Sequence | NDArray | pl.Series
EmbeddingVector: TypeAlias = Sequence[float] | NDArray | pl.Series

IntegerLabelList: TypeAlias = Sequence[int] | NDArray | pl.Series
IntegerLabelLists: TypeAlias = Sequence[IntegerLabelList]

StringLabelList: TypeAlias = Sequence[str] | pl.Series
StringLabelLists: TypeAlias = Sequence[StringLabelList] | pl.Series
