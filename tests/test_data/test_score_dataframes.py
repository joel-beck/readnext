import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import DocumentInfo, DocumentScore
from readnext.utils import ScoresFrame

score_dataframes: list[str] = [
    "co_citation_analysis_scores_most_cited",
    "bibliographic_coupling_scores_most_cited",
    "tfidf_cosine_similarities_most_cited",
    "bm25_cosine_similarities_most_cited",
    "word2vec_cosine_similarities_most_cited",
    "glove_cosine_similarities_most_cited",
    "fasttext_cosine_similarities_most_cited",
    "bert_cosine_similarities_most_cited",
    "scibert_cosine_similarities_most_cited",
    "longformer_cosine_similarities_most_cited",
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
]


@pytest.mark.slow
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

    # check document scores column
    first_document_scores: list[DocumentScore] = score_dataframe["scores"].iloc[0]
    assert isinstance(first_document_scores, list)

    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(score_dataframe) - 1)
        for document_scores in score_dataframe["scores"]
    )

    unique_index_ids = set(score_dataframe.index.tolist())
    unique_document_ids = {
        document_score.document_info.d3_document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1

    # check data types of document scores
    first_document_score: DocumentScore = first_document_scores[0]
    assert isinstance(first_document_score, DocumentScore)

    first_document_info: DocumentInfo = first_document_score.document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.d3_document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []
