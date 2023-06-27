import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import EmbeddingsFrame

keyword_algorithm_embedding_frames = [
    lazy_fixture("test_tfidf_embeddings_frame"),
    lazy_fixture("test_bm25_embeddings_frame"),
]
gensim_embedding_frames = [
    lazy_fixture("test_word2vec_embeddings_frame"),
    lazy_fixture("test_glove_embeddings_frame"),
    lazy_fixture("test_fasttext_embeddings_frame"),
]
torch_embedding_frames = [
    lazy_fixture("test_bert_embeddings_frame"),
    lazy_fixture("test_scibert_embeddings_frame"),
    lazy_fixture("test_longformer_embeddings_frame"),
]

all_embedding_frames = (
    keyword_algorithm_embedding_frames + gensim_embedding_frames + torch_embedding_frames
)


@pytest.mark.parametrize("embeddings_frame", all_embedding_frames)
def test_embeddings_frame_structure(embeddings_frame: EmbeddingsFrame) -> None:
    assert isinstance(embeddings_frame, pl.DataFrame)

    # check number and names of columns and index
    assert embeddings_frame.shape[1] == 2
    assert embeddings_frame.columns == ["d3_document_id", "embedding"]
    # check that `embedding` column contains lists of floats
    assert embeddings_frame.dtypes == [pl.Int64, pl.List(pl.Float64)]


@pytest.mark.parametrize("embeddings_frame", keyword_algorithm_embedding_frames)
def test_keyword_algorithm_embeddings_dimension(embeddings_frame: EmbeddingsFrame) -> None:
    # embedding dimension corresponds to size of corpus vocabulary
    assert all(len(embedding) == 21264 for embedding in embeddings_frame["embedding"])


@pytest.mark.parametrize("embeddings_frame", gensim_embedding_frames)
def test_gensim_embeddings_dimension(embeddings_frame: EmbeddingsFrame) -> None:
    assert all(len(embedding) == 300 for embedding in embeddings_frame["embedding"])


@pytest.mark.parametrize("embeddings_frame", torch_embedding_frames)
def test_torch_embeddings_dimension(embeddings_frame: EmbeddingsFrame) -> None:
    assert all(len(embedding) == 768 for embedding in embeddings_frame["embedding"])
