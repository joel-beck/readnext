import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import TokensIdMapping, TokensMapping
from readnext.utils import slice_mapping

real_test_tokens_mapping_pairs_skip_ci = [
    (
        "spacy_tokenized_abstracts_mapping_most_cited",
        "test_spacy_tokenized_abstracts_mapping_most_cited",
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
    test_data_size: int, real_tokens_mapping: TokensMapping, test_tokens_mapping: TokensMapping
) -> None:
    assert slice_mapping(real_tokens_mapping, test_data_size) == test_tokens_mapping


real_test_tokens_id_mapping_pairs_skip_ci = [
    (
        "bert_tokenized_abstracts_mapping_most_cited",
        "test_bert_tokenized_abstracts_mapping_most_cited",
    ),
    (
        "scibert_tokenized_abstracts_mapping_most_cited",
        "test_scibert_tokenized_abstracts_mapping_most_cited",
    ),
    (
        "longformer_tokenized_abstracts_mapping_most_cited",
        "test_longformer_tokenized_abstracts_mapping_most_cited",
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
    real_tokens_id_mapping: TokensIdMapping,
    test_tokens_id_mapping: TokensIdMapping,
) -> None:
    assert slice_mapping(real_tokens_id_mapping, test_data_size) == test_tokens_id_mapping


real_test_dataframe_pairs_skip_ci = [
    ("tfidf_embeddings_most_cited", "test_tfidf_embeddings_most_cited"),
    ("word2vec_embeddings_most_cited", "test_word2vec_embeddings_most_cited"),
    ("fasttext_embeddings_most_cited", "test_fasttext_embeddings_most_cited"),
    ("bert_embeddings_most_cited", "test_bert_embeddings_most_cited"),
    ("scibert_embeddings_most_cited", "test_scibert_embeddings_most_cited"),
    ("longformer_embeddings_most_cited", "test_longformer_embeddings_most_cited"),
    (
        "documents_authors_labels_citations_most_cited",
        "test_documents_authors_labels_citations_most_cited",
    ),
    ("co_citation_analysis_scores_most_cited", "test_co_citation_analysis_scores_most_cited"),
    (
        "bibliographic_coupling_scores_most_cited",
        "test_bibliographic_coupling_scores_most_cited",
    ),
    ("tfidf_cosine_similarities_most_cited", "test_tfidf_cosine_similarities_most_cited"),
    ("bm25_cosine_similarities_most_cited", "test_bm25_cosine_similarities_most_cited"),
    ("word2vec_cosine_similarities_most_cited", "test_word2vec_cosine_similarities_most_cited"),
    ("glove_cosine_similarities_most_cited", "test_glove_cosine_similarities_most_cited"),
    ("fasttext_cosine_similarities_most_cited", "test_fasttext_cosine_similarities_most_cited"),
    ("bert_cosine_similarities_most_cited", "test_bert_cosine_similarities_most_cited"),
    ("scibert_cosine_similarities_most_cited", "test_scibert_cosine_similarities_most_cited"),
    (
        "longformer_cosine_similarities_most_cited",
        "test_longformer_cosine_similarities_most_cited",
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
