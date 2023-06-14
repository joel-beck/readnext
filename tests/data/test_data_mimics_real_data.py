import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

real_test_dataframe_pairs_skip_ci = [
    (lazy_fixture("spacy_tokenized_abstracts"), lazy_fixture("test_spacy_tokenized_abstracts")),
    (lazy_fixture("bert_tokenized_abstracts"), lazy_fixture("test_bert_tokenized_abstracts")),
    (lazy_fixture("scibert_tokenized_abstracts"), lazy_fixture("test_scibert_tokenized_abstracts")),
    (
        lazy_fixture("longformer_tokenized_abstracts"),
        lazy_fixture("test_longformer_tokenized_abstracts"),
    ),
    (lazy_fixture("tfidf_embeddings"), lazy_fixture("test_tfidf_embeddings")),
    (lazy_fixture("word2vec_embeddings"), lazy_fixture("test_word2vec_embeddings")),
    (lazy_fixture("fasttext_embeddings"), lazy_fixture("test_fasttext_embeddings")),
    (lazy_fixture("bert_embeddings"), lazy_fixture("test_bert_embeddings")),
    (lazy_fixture("scibert_embeddings"), lazy_fixture("test_scibert_embeddings")),
    (lazy_fixture("longformer_embeddings"), lazy_fixture("test_longformer_embeddings")),
    (
        lazy_fixture("documents_authors_labels_citations"),
        lazy_fixture("test_documents_authors_labels_citations"),
    ),
    (lazy_fixture("co_citation_analysis_scores"), lazy_fixture("test_co_citation_analysis_scores")),
    (
        lazy_fixture("bibliographic_coupling_scores"),
        lazy_fixture("test_bibliographic_coupling_scores"),
    ),
    (lazy_fixture("tfidf_cosine_similarities"), lazy_fixture("test_tfidf_cosine_similarities")),
    (lazy_fixture("bm25_cosine_similarities"), lazy_fixture("test_bm25_cosine_similarities")),
    (
        lazy_fixture("word2vec_cosine_similarities"),
        lazy_fixture("test_word2vec_cosine_similarities"),
    ),
    (lazy_fixture("glove_cosine_similarities"), lazy_fixture("test_glove_cosine_similarities")),
    (
        lazy_fixture("fasttext_cosine_similarities"),
        lazy_fixture("test_fasttext_cosine_similarities"),
    ),
    (lazy_fixture("bert_cosine_similarities"), lazy_fixture("test_bert_cosine_similarities")),
    (lazy_fixture("scibert_cosine_similarities"), lazy_fixture("test_scibert_cosine_similarities")),
    (
        lazy_fixture("longformer_cosine_similarities"),
        lazy_fixture("test_longformer_cosine_similarities"),
    ),
]


@pytest.mark.updated
@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(("real_dataframe", "test_dataframe"), real_test_dataframe_pairs_skip_ci)
def test_that_test_dataframes_mimic_real_dataframes(
    test_data_size: int, real_dataframe: pl.DataFrame, test_dataframe: pl.DataFrame
) -> None:
    assert_frame_equal(real_dataframe.head(test_data_size), test_dataframe)
