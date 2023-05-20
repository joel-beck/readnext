from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from readnext.modeling.document_info import DocumentInfo

# each document is represented as a list of tokens
Tokens: TypeAlias = list[str]
TokensMapping: TypeAlias = dict[int, Tokens]

# each document is represented as a tensor of token ids
TokenIds: TypeAlias = list[int]
TokensIdMapping: TypeAlias = dict[int, list[int]]
TokensTensorMapping: TypeAlias = dict[int, torch.Tensor]

# tfidf and bm25
KeywordAlgorithm: TypeAlias = Callable[[Tokens, Sequence[Tokens]], np.ndarray]

Embedding: TypeAlias = NDArray
EmbeddingsMapping: TypeAlias = dict[int, Embedding]

# language model embedding functions during inference
QueryEmbeddingFunction: TypeAlias = Callable[[DocumentInfo], Embedding]

Vector: TypeAlias = Sequence | NDArray | pd.Series
EmbeddingVector: TypeAlias = Sequence[float] | NDArray | pd.Series

# data frame with a single column named `score` and an index named `document_id`
# the `score` column contains lists of `DocumentScore` objects
# used for storing co-citation analysis scores, bibliographic coupling scores and cosine
# similarity scores
ScoresFrame = pd.DataFrame

IntegerLabelList: TypeAlias = Sequence[int] | NDArray | pd.Series
IntegerLabelLists: TypeAlias = Sequence[IntegerLabelList]

StringLabelList: TypeAlias = Sequence[str] | pd.Series
StringLabelLists: TypeAlias = Sequence[StringLabelList] | pd.Series