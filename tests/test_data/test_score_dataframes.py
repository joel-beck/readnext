import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import DocumentInfo, DocumentScore
from readnext.utils import ScoresFrame

seen_score_dataframes = [
    "co_citation_analysis_scores",
    "bibliographic_coupling_scores",
    "tfidf_cosine_similarities",
    "bm25_cosine_similarities",
    "word2vec_cosine_similarities",
    "glove_cosine_similarities",
    "fasttext_cosine_similarities",
    "bert_cosine_similarities",
    "scibert_cosine_similarities",
    "longformer_cosine_similarities",
    "seen_paper_attribute_getter_co_citation_analysis",
    "seen_paper_attribute_getter_bibliographic_coupling",
    "seen_paper_attribute_getter_cosine_similarities_tfidf",
    "seen_paper_attribute_getter_cosine_similarities_bm25",
    "seen_paper_attribute_getter_cosine_similarities_word2vec",
    "seen_paper_attribute_getter_cosine_similarities_glove",
    "seen_paper_attribute_getter_cosine_similarities_fasttext",
    "seen_paper_attribute_getter_cosine_similarities_bert",
    "seen_paper_attribute_getter_cosine_similarities_scibert",
    "seen_paper_attribute_getter_cosine_similarities_longformer",
    "inference_data_constructor_seen_co_citation_analysis_scores",
    "inference_data_constructor_seen_bibliographic_coupling_scores",
    "inference_data_constructor_seen_cosine_similarities",
]

unseen_score_dataframes = [
    "unseen_paper_attribute_getter_co_citation_analysis",
    "unseen_paper_attribute_getter_bibliographic_coupling",
    "unseen_paper_attribute_getter_cosine_similarities_tfidf",
    "unseen_paper_attribute_getter_cosine_similarities_bm25",
    "unseen_paper_attribute_getter_cosine_similarities_word2vec",
    "unseen_paper_attribute_getter_cosine_similarities_glove",
    "unseen_paper_attribute_getter_cosine_similarities_fasttext",
    "unseen_paper_attribute_getter_cosine_similarities_bert",
    "unseen_paper_attribute_getter_cosine_similarities_scibert",
    "unseen_paper_attribute_getter_cosine_similarities_longformer",
    "inference_data_constructor_unseen_co_citation_analysis_scores",
    "inference_data_constructor_unseen_bibliographic_coupling_scores",
    "inference_data_constructor_unseen_cosine_similarities",
]

score_dataframes = seen_score_dataframes + unseen_score_dataframes


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "score_dataframe",
    lazy_fixture(score_dataframes),
)
def test_score_dataframes(
    score_dataframe: ScoresFrame,
) -> None:
    assert isinstance(score_dataframe, pd.DataFrame)

    # check number and names of columns and index
    assert score_dataframe.shape[1] == 1
    assert score_dataframe.index.name == "document_id"
    assert score_dataframe.columns.tolist() == ["scores"]

    # check document id data type
    assert is_integer_dtype(score_dataframe.index)

    # check first document score
    first_document_scores: list[DocumentScore] = score_dataframe["scores"].iloc[0]
    assert isinstance(first_document_scores, list)
    assert all(
        isinstance(document_score, DocumentScore) for document_score in first_document_scores
    )

    # check that document scores are ordered by their score in descending order
    first_document_scores_sorted = sorted(
        first_document_scores, key=lambda x: x.score, reverse=True
    )
    assert first_document_scores == first_document_scores_sorted

    # check first document score content
    first_document_info: DocumentInfo = first_document_scores[0].document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.d3_document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "score_dataframe",
    lazy_fixture(seen_score_dataframes),
)
def test_seen_score_dataframes(
    score_dataframe: ScoresFrame,
) -> None:
    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(score_dataframe) - 1)
        for document_scores in score_dataframe["scores"]
    )

    # check document scores column
    first_document_scores: list[DocumentScore] = score_dataframe["scores"].iloc[0]

    unique_index_ids = set(score_dataframe.index.tolist())
    unique_document_ids = {
        document_score.document_info.d3_document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "score_dataframe",
    lazy_fixture(unseen_score_dataframes),
)
def test_unseen_score_dataframes(
    score_dataframe: ScoresFrame,
) -> None:
    # check that scores dataframe contains only a single row with index -1 for the
    # unseen query document
    assert len(score_dataframe) == 1
    assert score_dataframe.index.tolist() == [-1]
