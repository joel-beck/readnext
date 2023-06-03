import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

real_test_dataframe_pairs_skip_ci = [
    ("spacy_tokenized_abstracts", "test_spacy_tokenized_abstracts"),
    ("bert_tokenized_abstracts", "test_bert_tokenized_abstracts"),
    ("scibert_tokenized_abstracts", "test_scibert_tokenized_abstracts"),
    ("longformer_tokenized_abstracts", "test_longformer_tokenized_abstracts"),
    ("tfidf_embeddings", "test_tfidf_embeddings"),
    ("word2vec_embeddings", "test_word2vec_embeddings"),
    ("fasttext_embeddings", "test_fasttext_embeddings"),
    ("bert_embeddings", "test_bert_embeddings"),
    ("scibert_embeddings", "test_scibert_embeddings"),
    ("longformer_embeddings", "test_longformer_embeddings"),
    ("documents_authors_labels_citations", "test_documents_authors_labels_citations"),
    ("co_citation_analysis_scores", "test_co_citation_analysis_scores"),
    ("bibliographic_coupling_scores", "test_bibliographic_coupling_scores"),
    ("tfidf_cosine_similarities", "test_tfidf_cosine_similarities"),
    ("bm25_cosine_similarities", "test_bm25_cosine_similarities"),
    ("word2vec_cosine_similarities", "test_word2vec_cosine_similarities"),
    ("glove_cosine_similarities", "test_glove_cosine_similarities"),
    ("fasttext_cosine_similarities", "test_fasttext_cosine_similarities"),
    ("bert_cosine_similarities", "test_bert_cosine_similarities"),
    ("scibert_cosine_similarities", "test_scibert_cosine_similarities"),
    ("longformer_cosine_similarities", "test_longformer_cosine_similarities"),
]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    ("real_dataframe", "test_dataframe"),
    [
        (lazy_fixture(real_dataframe), lazy_fixture(test_dataframe))
        for real_dataframe, test_dataframe in real_test_dataframe_pairs_skip_ci
    ],
)
def test_that_test_dataframes_mimic_real_dataframes(
    test_data_size: int, real_dataframe: pl.DataFrame, test_dataframe: pl.DataFrame
) -> None:
    assert_frame_equal(real_dataframe.head(test_data_size), test_dataframe)
