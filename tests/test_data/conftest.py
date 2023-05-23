import pandas as pd
import pytest
from pytest_mock import MockFixture

from readnext.config import DataPaths, ResultsPaths
from readnext.data import SemanticScholarJson, SemanticscholarRequest, SemanticScholarResponse
from readnext.modeling.language_models import TokensIdMapping, TokensMapping
from readnext.utils import load_df_from_pickle, load_object_from_pickle


# SECTION: Local Data
@pytest.fixture(scope="session")
def documents_authors_labels_citations_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(DataPaths.merged.documents_authors_labels_citations_most_cited_pkl)


@pytest.fixture(scope="session")
def spacy_tokenized_abstracts_mapping_most_cited() -> TokensMapping:
    return load_object_from_pickle(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="session")
def bert_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return load_object_from_pickle(
        ResultsPaths.language_models.bert_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="session")
def scibert_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return load_object_from_pickle(
        ResultsPaths.language_models.scibert_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="session")
def longformer_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return load_object_from_pickle(
        ResultsPaths.language_models.longformer_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="session")
def tfidf_embeddings_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(ResultsPaths.language_models.tfidf_embeddings_most_cited_pkl)


@pytest.fixture(scope="session")
def bm25_embeddings_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(ResultsPaths.language_models.bm25_embeddings_most_cited_pkl)


@pytest.fixture(scope="session")
def word2vec_embeddings_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(ResultsPaths.language_models.word2vec_embeddings_most_cited_pkl)


@pytest.fixture(scope="session")
def glove_embeddings_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(ResultsPaths.language_models.glove_embeddings_most_cited_pkl)


@pytest.fixture(scope="session")
def fasttext_embeddings_most_cited() -> pd.DataFrame:
    return load_object_from_pickle(ResultsPaths.language_models.fasttext_embeddings_most_cited_pkl)


@pytest.fixture(scope="session")
def bert_embeddings_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(ResultsPaths.language_models.bert_embeddings_most_cited_pkl)


@pytest.fixture(scope="session")
def scibert_embeddings_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(ResultsPaths.language_models.scibert_embeddings_most_cited_pkl)


@pytest.fixture(scope="session")
def longformer_embeddings_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(ResultsPaths.language_models.longformer_embeddings_most_cited_pkl)


@pytest.fixture(scope="session")
def co_citation_analysis_scores_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
    )


@pytest.fixture(scope="session")
def bibliographic_coupling_scores_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl
    )


@pytest.fixture(scope="session")
def tfidf_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
    )


@pytest.fixture(scope="session")
def bm25_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(ResultsPaths.language_models.bm25_cosine_similarities_most_cited_pkl)


@pytest.fixture(scope="session")
def word2vec_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
    )


@pytest.fixture(scope="session")
def glove_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.glove_cosine_similarities_most_cited_pkl
    )


@pytest.fixture(scope="session")
def fasttext_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
    )


@pytest.fixture(scope="session")
def bert_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl)


@pytest.fixture(scope="session")
def scibert_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
    )


@pytest.fixture(scope="session")
def longformer_cosine_similarities_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        ResultsPaths.language_models.longformer_cosine_similarities_most_cited_pkl
    )


# SECTION: Citation Models Features
@pytest.fixture
def citation_models_features_frame() -> pd.DataFrame:
    data = {
        "publication_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2019-01-01"), None],
        "citationcount_document": [50, 100, 75],
        "citationcount_author": [1000, 2000, 3000],
    }
    return pd.DataFrame(data)


@pytest.fixture
def extended_citation_models_features_frame() -> pd.DataFrame:
    data = {
        "publication_date": [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2019-01-01"),
            None,
            pd.Timestamp("2018-01-01"),
            pd.Timestamp("2017-01-01"),
        ],
        "citationcount_document": [50, 100, 75, 50, 100],
        "citationcount_author": [1000, 2000, 3000, 1000, 2000],
    }
    return pd.DataFrame(data)


# SECTION: Semanticscholar Class
@pytest.fixture
def semanticscholar_request() -> SemanticscholarRequest:
    return SemanticscholarRequest()


@pytest.fixture
def json_response() -> SemanticScholarJson:
    return {
        "paperId": "TestID",
        "title": "TestTitle",
        "abstract": "TestAbstract",
        "citations": [],
        "references": [],
        "externalIds": {"ArXiv": "ArxivID", "DBLP": None, "PubMedCentral": None},
    }


@pytest.fixture
def semanticscholar_response() -> SemanticScholarResponse:
    return SemanticScholarResponse(
        semanticscholar_id="TestID",
        arxiv_id="ArxivID",
        title="TestTitle",
        abstract="TestAbstract",
        citations=[],
        references=[],
    )


@pytest.fixture
def mock_get_response_from_request(
    mocker: MockFixture, semanticscholar_response: SemanticScholarResponse
) -> None:
    mocker.patch.object(
        SemanticscholarRequest, "get_response_from_request", return_value=semanticscholar_response
    )
