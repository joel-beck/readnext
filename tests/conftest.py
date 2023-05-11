from pathlib import Path

import pandas as pd
import pytest

from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    DocumentsInfo,
    LanguageModelDataConstructor,
)
from readnext.modeling.citation_models import (
    add_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling.language_models import Tokens
from readnext.utils import load_df_from_pickle, load_object_from_pickle


@pytest.fixture(scope="session")
def root_path() -> Path:
    """Return project root path when pytest is executed from the project root directory."""
    return Path().cwd()


@pytest.fixture(scope="session")
def test_data_size() -> int:
    return 1000


@pytest.fixture(scope="session")
def test_bert_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bert_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bert_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_tokenized_abstracts_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bert_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bibliographic_coupling_scores_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bibliographic_coupling_scores_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bm25_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bm25_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bm25_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bm25_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_co_citation_analysis_scores_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_co_citation_analysis_scores_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_documents_authors_labels_citations_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_documents_authors_labels_citations_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_fasttext_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_fasttext_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_fasttext_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_fasttext_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_glove_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_glove_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_glove_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_glove_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_scibert_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_scibert_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_tokenized_abstracts_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_scibert_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_spacy_tokenized_abstracts_mapping_most_cited(root_path: Path) -> pd.DataFrame:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_spacy_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_tfidf_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_tfidf_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_tfidf_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_tfidf_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_word2vec_cosine_similarities_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_word2vec_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_word2vec_embeddings_mapping_most_cited(root_path: Path) -> dict:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_word2vec_embeddings_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def documents_info() -> DocumentsInfo:
    return DocumentsInfo(
        [
            DocumentInfo(
                document_id=1,
                title="Title 1",
                author="Author 1",
                abstract="""
                Abstract 1: This is an example abstract with various characters! It
                contains numbers 1, 2, 3 and special characters like @, #, $.
                """,
            ),
            DocumentInfo(
                document_id=2,
                title="Title 2",
                author="Author 2",
                abstract="""
                Abstract 2: Another example abstract, including upper-case letters and a
                few stopwords such as 'the', 'and', 'in'.
                """,
            ),
            DocumentInfo(
                document_id=3,
                title="Title 3",
                author="Author 3",
                abstract="""
                Abstract 3: A third example abstract with a mix of lower-case and
                UPPER-CASE letters, as well as some punctuation: (brackets) and {curly
                braces}.
                """,
            ),
        ]
    )


@pytest.fixture(scope="module")
def citation_model_data_constructor(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
    test_co_citation_analysis_scores_most_cited: pd.DataFrame,
    test_bibliographic_coupling_scores_most_cited: pd.DataFrame,
) -> CitationModelDataConstructor:
    query_document_id = 546182

    return CitationModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=test_documents_authors_labels_citations_most_cited.pipe(
            add_feature_rank_cols
        ).pipe(set_missing_publication_dates_to_max_rank),
        co_citation_analysis_scores=test_co_citation_analysis_scores_most_cited,
        bibliographic_coupling_scores=test_bibliographic_coupling_scores_most_cited,
    )


@pytest.fixture(scope="module")
def language_model_data_constructor(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
    test_tfidf_cosine_similarities_most_cited: pd.DataFrame,
) -> LanguageModelDataConstructor:
    query_document_id = 546182

    return LanguageModelDataConstructor(
        query_document_id=query_document_id,
        documents_data=test_documents_authors_labels_citations_most_cited.pipe(
            add_feature_rank_cols
        ).pipe(set_missing_publication_dates_to_max_rank),
        cosine_similarities=test_tfidf_cosine_similarities_most_cited,
    )


@pytest.fixture
def citation_model_data_constructor_new_document_id(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> CitationModelDataConstructor:
    # original query document id is not in co-citation analysis scores and bibliographic
    # coupling scores data
    citation_model_data_constructor.query_document_id = 206594692
    return citation_model_data_constructor


@pytest.fixture
def language_model_data_constructor_new_document_id(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> LanguageModelDataConstructor:
    # original query document id is not in cosine similarity scores data
    language_model_data_constructor.query_document_id = 206594692
    return language_model_data_constructor


@pytest.fixture
def document_tokens() -> Tokens:
    return ["a", "b", "c", "a", "b", "c", "d", "d", "d"]


@pytest.fixture
def document_corpus() -> list[Tokens]:
    return [
        ["a", "b", "c", "d", "d", "d"],
        ["a", "b", "b", "c", "c", "c", "d"],
        ["a", "a", "a", "b", "c", "d"],
    ]
