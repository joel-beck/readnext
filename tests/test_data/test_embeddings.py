import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

keyword_algorithm_embeddings = ["tfidf_embeddings", "bm25_embeddings"]

gensim_embeddings = ["word2vec_embeddings", "glove_embeddings", "fasttext_embeddings"]

torch_embeddings = ["bert_embeddings", "scibert_embeddings", "longformer_embeddings"]

all_embeddings = keyword_algorithm_embeddings + gensim_embeddings + torch_embeddings


@pytest.mark.skip_ci
@pytest.mark.parametrize("embeddings", lazy_fixture(all_embeddings))
def test_embeddings_dataframe_structure(embeddings: pl.DataFrame) -> None:
    assert isinstance(embeddings, pl.DataFrame)

    # check number and names of columns and index
    assert embeddings.shape[1] == 2
    assert embeddings.columns == ["d3_document_id", "embedding"]
    # check that `embedding` column contains lists of floats
    assert embeddings.dtypes == [pl.Int64, list]
    assert all(
        isinstance(embedding_dimension, float) for embedding_dimension in embeddings["embedding"]
    )


@pytest.mark.skip_ci
@pytest.mark.parametrize("embeddings", lazy_fixture(keyword_algorithm_embeddings))
def test_keyword_algorithm_embeddings_dimension(embeddings: pl.DataFrame) -> None:
    # embedding dimension corresponds to size of corpus vocabulary
    assert all(len(embedding) == 6578 for embedding in embeddings["embedding"])


@pytest.mark.skip_ci
@pytest.mark.parametrize("embeddings", lazy_fixture(gensim_embeddings))
def test_gensim_embeddings_dimension(embeddings: pl.DataFrame) -> None:
    assert all(len(embedding) == 300 for embedding in embeddings["embedding"])


@pytest.mark.skip_ci
@pytest.mark.parametrize("embeddings", lazy_fixture(torch_embeddings))
def test_torch_embeddings_dimension(embeddings: pl.DataFrame) -> None:
    assert all(len(embedding) == 768 for embedding in embeddings["embedding"])
