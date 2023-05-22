import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from pytest_lazyfixture import lazy_fixture

keyword_algorithm_embeddings = ["tfidf_embeddings_most_cited", "bm25_embeddings_most_cited"]

gensim_embeddings = [
    "word2vec_embeddings_most_cited",
    "glove_embeddings_most_cited",
    "fasttext_embeddings_most_cited",
]

torch_embeddings = [
    "bert_embeddings_most_cited",
    "scibert_embeddings_most_cited",
    "longformer_embeddings_most_cited",
]

all_embeddings = keyword_algorithm_embeddings + gensim_embeddings + torch_embeddings


@pytest.mark.skip_ci
@pytest.mark.parametrize("embeddings", lazy_fixture(all_embeddings))
def test_embeddings_dataframe_structure(embeddings: pd.DataFrame) -> None:
    assert isinstance(embeddings, pd.DataFrame)

    # check number and names of columns and index
    assert embeddings.shape[1] == 1
    assert embeddings.index.name == "document_id"
    assert embeddings.columns.tolist() == ["embedding"]

    # check dtypes of index and columns
    assert is_integer_dtype(embeddings.index)
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings["embedding"])


@pytest.mark.skip_ci
@pytest.mark.parametrize("embeddings", lazy_fixture(keyword_algorithm_embeddings))
def test_keyword_algorithm_embeddings_dimension(embeddings: pd.DataFrame) -> None:
    first_document = embeddings.iloc[0]
    assert first_document["embedding"].dtype == np.float64
    assert all(len(embedding) == 6578 for embedding in embeddings["embedding"])


@pytest.mark.skip_ci
@pytest.mark.parametrize("embeddings", lazy_fixture(gensim_embeddings))
def test_gensim_embeddings_dimension(embeddings: pd.DataFrame) -> None:
    first_document = embeddings.iloc[0]
    assert first_document["embedding"].dtype == np.float32

    # check embedding dimension
    assert all(len(embedding) == 300 for embedding in embeddings["embedding"])


@pytest.mark.skip_ci
@pytest.mark.parametrize("embeddings", lazy_fixture(torch_embeddings))
def test_torch_embeddings_dimension(embeddings: pd.DataFrame) -> None:
    first_document = embeddings.iloc[0]
    assert first_document["embedding"].dtype == np.float32

    # check embedding dimension
    assert all(len(embedding) == 768 for embedding in embeddings["embedding"])
