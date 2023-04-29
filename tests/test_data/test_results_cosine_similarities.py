import pytest
import pandas as pd

from readnext.config import ResultsPaths
from readnext.modeling import DocumentScore, DocumentInfo
from readnext.utils import load_df_from_pickle
from pandas.api.types import is_integer_dtype


@pytest.fixture(scope="module")
def tfidf_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
    )


@pytest.fixture(scope="module")
def word2vec_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
    )


@pytest.fixture(scope="module")
def fasttext_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )


@pytest.fixture(scope="module")
def bert_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl)


@pytest.fixture(scope="module")
def scibert_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
    )


def test_tfidf_cosine_similarities_most_cited(
    tfidf_cosine_similarities_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(tfidf_cosine_similarities_most_cited, pd.DataFrame)

    # check number and names of columns and index
    assert tfidf_cosine_similarities_most_cited.shape[1] == 1
    assert tfidf_cosine_similarities_most_cited.index.name == "document_id"
    assert tfidf_cosine_similarities_most_cited.columns.tolist() == ["scores"]

    # check document id data type
    assert is_integer_dtype(tfidf_cosine_similarities_most_cited.index)

    # check document scores column
    first_document_scores: list[DocumentScore] = tfidf_cosine_similarities_most_cited[
        "scores"
    ].iloc[0]
    assert isinstance(first_document_scores, list)

    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(tfidf_cosine_similarities_most_cited) - 1)
        for document_scores in tfidf_cosine_similarities_most_cited["scores"]
    )

    unique_index_ids = set(tfidf_cosine_similarities_most_cited.index.tolist())
    unique_document_ids = {
        document_score.document_info.document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1

    # check data types of document scores
    first_document_score: DocumentScore = first_document_scores[0]
    assert isinstance(first_document_score, DocumentScore)

    first_document_info: DocumentInfo = first_document_score.document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []


def test_word2vec_cosine_similarities_most_cited(
    word2vec_cosine_similarities_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(word2vec_cosine_similarities_most_cited, pd.DataFrame)

    # check number and names of columns and index
    assert word2vec_cosine_similarities_most_cited.shape[1] == 1
    assert word2vec_cosine_similarities_most_cited.index.name == "document_id"
    assert word2vec_cosine_similarities_most_cited.columns.tolist() == ["scores"]

    # check document id data type
    assert is_integer_dtype(word2vec_cosine_similarities_most_cited.index)

    # check document scores column
    first_document_scores: list[DocumentScore] = word2vec_cosine_similarities_most_cited[
        "scores"
    ].iloc[0]
    assert isinstance(first_document_scores, list)

    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(word2vec_cosine_similarities_most_cited) - 1)
        for document_scores in word2vec_cosine_similarities_most_cited["scores"]
    )

    unique_index_ids = set(word2vec_cosine_similarities_most_cited.index.tolist())
    unique_document_ids = {
        document_score.document_info.document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1

    # check data types of document scores
    first_document_score: DocumentScore = first_document_scores[0]
    assert isinstance(first_document_score, DocumentScore)

    first_document_info: DocumentInfo = first_document_score.document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []


def test_fasttext_cosine_similarities_most_cited(
    fasttext_cosine_similarities_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(fasttext_cosine_similarities_most_cited, pd.DataFrame)

    # check number and names of columns and index
    assert fasttext_cosine_similarities_most_cited.shape[1] == 1
    assert fasttext_cosine_similarities_most_cited.index.name == "document_id"
    assert fasttext_cosine_similarities_most_cited.columns.tolist() == ["scores"]

    # check document id data type
    assert is_integer_dtype(fasttext_cosine_similarities_most_cited.index)

    # check document scores column
    first_document_scores: list[DocumentScore] = fasttext_cosine_similarities_most_cited[
        "scores"
    ].iloc[0]
    assert isinstance(first_document_scores, list)

    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(fasttext_cosine_similarities_most_cited) - 1)
        for document_scores in fasttext_cosine_similarities_most_cited["scores"]
    )

    unique_index_ids = set(fasttext_cosine_similarities_most_cited.index.tolist())
    unique_document_ids = {
        document_score.document_info.document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1

    # check data types of document scores
    first_document_score: DocumentScore = first_document_scores[0]
    assert isinstance(first_document_score, DocumentScore)

    first_document_info: DocumentInfo = first_document_score.document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []


def test_bert_cosine_similarities_most_cited(
    bert_cosine_similarities_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(bert_cosine_similarities_most_cited, pd.DataFrame)

    # check number and names of columns and index
    assert bert_cosine_similarities_most_cited.shape[1] == 1
    assert bert_cosine_similarities_most_cited.index.name == "document_id"
    assert bert_cosine_similarities_most_cited.columns.tolist() == ["scores"]

    # check document id data type
    assert is_integer_dtype(bert_cosine_similarities_most_cited.index)

    # check document scores column
    first_document_scores: list[DocumentScore] = bert_cosine_similarities_most_cited["scores"].iloc[
        0
    ]
    assert isinstance(first_document_scores, list)

    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(bert_cosine_similarities_most_cited) - 1)
        for document_scores in bert_cosine_similarities_most_cited["scores"]
    )

    unique_index_ids = set(bert_cosine_similarities_most_cited.index.tolist())
    unique_document_ids = {
        document_score.document_info.document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1

    # check data types of document scores
    first_document_score: DocumentScore = first_document_scores[0]
    assert isinstance(first_document_score, DocumentScore)

    first_document_info: DocumentInfo = first_document_score.document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []


def test_scibert_cosine_similarities_most_cited(
    scibert_cosine_similarities_most_cited: pd.DataFrame,
) -> None:
    assert isinstance(scibert_cosine_similarities_most_cited, pd.DataFrame)

    # check number and names of columns and index
    assert scibert_cosine_similarities_most_cited.shape[1] == 1
    assert scibert_cosine_similarities_most_cited.index.name == "document_id"
    assert scibert_cosine_similarities_most_cited.columns.tolist() == ["scores"]

    # check document id data type
    assert is_integer_dtype(scibert_cosine_similarities_most_cited.index)

    # check document scores column
    first_document_scores: list[DocumentScore] = scibert_cosine_similarities_most_cited[
        "scores"
    ].iloc[0]
    assert isinstance(first_document_scores, list)

    # check that scores for all documents in training corpus are present except for the
    # query document
    assert all(
        len(document_scores) == (len(scibert_cosine_similarities_most_cited) - 1)
        for document_scores in scibert_cosine_similarities_most_cited["scores"]
    )

    unique_index_ids = set(scibert_cosine_similarities_most_cited.index.tolist())
    unique_document_ids = {
        document_score.document_info.document_id for document_score in first_document_scores
    }
    assert len(unique_index_ids - unique_document_ids) == 1

    # check data types of document scores
    first_document_score: DocumentScore = first_document_scores[0]
    assert isinstance(first_document_score, DocumentScore)

    first_document_info: DocumentInfo = first_document_score.document_info
    assert isinstance(first_document_info, DocumentInfo)

    # check that only document_id of document_info is set
    assert isinstance(first_document_info.document_id, int)
    assert first_document_info.title == ""
    assert first_document_info.author == ""
    assert first_document_info.abstract == ""
    assert first_document_info.arxiv_labels == []
