import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture
from readnext.utils import TokensFrame, TokenIdsFrame

real_test_tokens_mapping_pairs_skip_ci = [
    (
        "spacy_tokenized_abstracts_mapping",
        "test_spacy_tokenized_abstracts_mapping",
    ),
]


@pytest.mark.skip_ci
@pytest.mark.parametrize(
    ("real_tokens_mapping", "test_tokens_mapping"),
    [
        (lazy_fixture(real_tokens_mapping), lazy_fixture(test_tokens_mapping))
        for real_tokens_mapping, test_tokens_mapping in real_test_tokens_mapping_pairs_skip_ci
    ],
)
def test_that_test_tokens_mappings_mimic_real_tokens_mappings(
    test_data_size: int, real_tokens_mapping: TokensFrame, test_tokens_mapping: TokensFrame
) -> None:
    assert real_tokens_mapping.head(test_data_size) == test_tokens_mapping


real_test_tokens_id_mapping_pairs_skip_ci = [
    (
        "bert_tokenized_abstracts_mapping",
        "test_bert_tokenized_abstracts_mapping",
    ),
    (
        "scibert_tokenized_abstracts_mapping",
        "test_scibert_tokenized_abstracts_mapping",
    ),
    (
        "longformer_tokenized_abstracts_mapping",
        "test_longformer_tokenized_abstracts_mapping",
    ),
]


@pytest.mark.skip_ci
@pytest.mark.parametrize(
    ("real_tokens_id_mapping", "test_tokens_id_mapping"),
    [
        (lazy_fixture(real_tokens_id_mapping), lazy_fixture(test_tokens_id_mapping))
        for real_tokens_id_mapping, test_tokens_id_mapping in real_test_tokens_id_mapping_pairs_skip_ci  # noqa: E501
    ],
)
def test_that_test_tokens_id_mappings_mimic_real_tokens_id_mappings(
    test_data_size: int,
    real_tokens_id_mapping: TokenIdsFrame,
    test_tokens_id_mapping: TokenIdsFrame,
) -> None:
    assert real_tokens_id_mapping.head(test_data_size) == test_tokens_id_mapping


real_test_dataframe_pairs_skip_ci = [
    ("tfidf_embeddings", "test_tfidf_embeddings"),
    ("word2vec_embeddings", "test_word2vec_embeddings"),
    ("fasttext_embeddings", "test_fasttext_embeddings"),
    ("bert_embeddings", "test_bert_embeddings"),
    ("scibert_embeddings", "test_scibert_embeddings"),
    ("longformer_embeddings", "test_longformer_embeddings"),
    (
        "documents_authors_labels_citations",
        "test_documents_authors_labels_citations",
    ),
    ("co_citation_analysis_scores", "test_co_citation_analysis_scores"),
    (
        "bibliographic_coupling_scores",
        "test_bibliographic_coupling_scores",
    ),
    ("tfidf_cosine_similarities", "test_tfidf_cosine_similarities"),
    ("bm25_cosine_similarities", "test_bm25_cosine_similarities"),
    ("word2vec_cosine_similarities", "test_word2vec_cosine_similarities"),
    ("glove_cosine_similarities", "test_glove_cosine_similarities"),
    ("fasttext_cosine_similarities", "test_fasttext_cosine_similarities"),
    ("bert_cosine_similarities", "test_bert_cosine_similarities"),
    ("scibert_cosine_similarities", "test_scibert_cosine_similarities"),
    (
        "longformer_cosine_similarities",
        "test_longformer_cosine_similarities",
    ),
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
    test_data_size: int, real_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame
) -> None:
    assert_frame_equal(real_dataframe.head(test_data_size), test_dataframe)
