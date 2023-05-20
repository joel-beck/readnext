import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelData,
    LanguageModelDataConstructor,
    ModelData,
)

citation_model_data_constructor_fixtures = ["citation_model_data_constructor"]
language_model_data_constructor_fixtures = ["language_model_data_constructor"]
model_data_constructor_fixtures = (
    citation_model_data_constructor_fixtures + language_model_data_constructor_fixtures
)

citation_model_data_fixtures = [
    "citation_model_data",
    "seen_paper_attribute_getter_citation_model_data",
]

language_model_data_fixtures = [
    "language_model_data",
    "seen_paper_attribute_getter_language_model_data",
]

model_data_fixtures = citation_model_data_fixtures + language_model_data_fixtures


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(citation_model_data_constructor_fixtures),
)
def test_citation_model_data_from_constructor(
    model_data_constructor: CitationModelDataConstructor,
) -> None:
    citation_model_data = CitationModelData.from_constructor(model_data_constructor)

    assert isinstance(citation_model_data, CitationModelData)
    assert isinstance(citation_model_data.info_matrix, pd.DataFrame)
    assert isinstance(citation_model_data.integer_labels, pd.Series)
    assert isinstance(citation_model_data.feature_matrix, pd.DataFrame)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(language_model_data_constructor_fixtures),
)
def test_language_model_data_from_constructor(
    model_data_constructor: LanguageModelDataConstructor,
) -> None:
    language_model_data = LanguageModelData.from_constructor(model_data_constructor)

    assert isinstance(language_model_data, LanguageModelData)
    assert isinstance(language_model_data.info_matrix, pd.DataFrame)
    assert isinstance(language_model_data.integer_labels, pd.Series)
    assert isinstance(language_model_data.cosine_similarity_ranks, pd.DataFrame)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data",
    lazy_fixture(model_data_fixtures),
)
def test_model_data_query_document(model_data: ModelData) -> None:
    assert isinstance(model_data.query_document, DocumentInfo)

    # TODO: For model data the index if of type int (the python type) and not int64 (the
    # numpy type)
    assert model_data.query_document.d3_document_id.dtype == np.int64  # type: ignore
    assert model_data.query_document.d3_document_id == 206594692

    assert isinstance(model_data.query_document.title, str)
    assert model_data.query_document.title == "Deep Residual Learning for Image Recognition"

    assert isinstance(model_data.query_document.author, str)
    assert model_data.query_document.author == "Kaiming He"

    assert isinstance(model_data.query_document.arxiv_labels, list)
    assert all(isinstance(label, str) for label in model_data.query_document.arxiv_labels)
    assert model_data.query_document.arxiv_labels == ["cs.CV"]

    assert isinstance(model_data.query_document.abstract, str)
    assert model_data.query_document.abstract == ""


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data",
    lazy_fixture(model_data_fixtures),
)
def test_model_data_integer_labels(model_data: ModelData) -> None:
    assert isinstance(model_data.integer_labels, pd.Series)

    assert model_data.integer_labels.dtype == np.int64  # type: ignore
    assert model_data.integer_labels.name == "integer_labels"
    assert model_data.integer_labels.index.name == "document_id"

    assert model_data.integer_labels.unique().tolist() == [0, 1]


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data",
    lazy_fixture(model_data_fixtures),
)
def test_model_data_info_matrix(model_data: ModelData) -> None:
    assert isinstance(model_data.info_matrix, pd.DataFrame)

    assert model_data.info_matrix.index.name == "document_id"
    assert model_data.info_matrix.index.dtype == pd.Int64Dtype()

    assert model_data.info_matrix.shape[1] == 8
    assert model_data.info_matrix.columns.to_list() == [
        "title",
        "author",
        "arxiv_labels",
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis",
        "bibliographic_coupling",
    ]
    # TODO: This test fails
    assert all(
        model_data.info_matrix.dtypes
        == [
            pd.StringDtype(),
            pd.StringDtype(),
            object,
            pd.StringDtype(),
            pd.Int64Dtype(),
            pd.Int64Dtype(),
            np.float64,
            np.float64,
        ]
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data",
    lazy_fixture(citation_model_data_fixtures),
)
def test_citation_model_data_feature_matrix(model_data: CitationModelData) -> None:
    assert isinstance(model_data.feature_matrix, pd.DataFrame)

    assert model_data.feature_matrix.index.name == "document_id"
    assert model_data.feature_matrix.index.dtype == pd.Int64Dtype()

    assert model_data.feature_matrix.shape[1] == 5
    assert model_data.feature_matrix.columns.to_list() == [
        "publication_date_rank",
        "citationcount_document_rank",
        "citationcount_author_rank",
        "co_citation_analysis_rank",
        "bibliographic_coupling_rank",
    ]
    assert all(
        model_data.feature_matrix.dtypes
        == [np.float64, np.float64, np.float64, np.float64, np.float64]
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data",
    lazy_fixture(language_model_data_fixtures),
)
def test_language_model_data_cosine_similarity_ranks(
    model_data: LanguageModelData,
) -> None:
    assert isinstance(model_data.cosine_similarity_ranks, pd.DataFrame)

    assert model_data.cosine_similarity_ranks.index.name == "document_id"
    assert model_data.cosine_similarity_ranks.index.dtype == np.int64

    assert model_data.cosine_similarity_ranks.shape[1] == 1
    assert model_data.cosine_similarity_ranks.columns.to_list() == ["cosine_similarity_rank"]
    assert all(model_data.cosine_similarity_ranks.dtypes == [np.float64])

    # check that lowest / best rank is 1.0
    assert min(model_data.cosine_similarity_ranks["cosine_similarity_rank"]) == 1.0

    # check that no rank is higher than the number of documents
    assert max(model_data.cosine_similarity_ranks["cosine_similarity_rank"]) < len(
        model_data.cosine_similarity_ranks
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data",
    lazy_fixture(citation_model_data_fixtures),
)
def test_citation_model_data_getitem(model_data: CitationModelData, test_data_size: int) -> None:
    index_info_matrix = model_data.info_matrix.index
    assert len(index_info_matrix) == test_data_size - 1

    index_feature_matrix = model_data.feature_matrix.index
    assert len(index_feature_matrix) == test_data_size - 1

    # index of info matrix and feature matrix is identical
    shared_indices = index_info_matrix.intersection(index_feature_matrix)
    assert len(shared_indices) == test_data_size - 1

    # check that slicing works for info matrix, feature matrix and integer labels
    sliced_model_data = model_data[shared_indices]
    assert isinstance(sliced_model_data, CitationModelData)

    assert len(sliced_model_data.info_matrix) == len(shared_indices)
    assert len(sliced_model_data.integer_labels) == len(shared_indices)
    assert len(sliced_model_data.feature_matrix) == len(shared_indices)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_data",
    lazy_fixture(language_model_data_fixtures),
)
def test_language_model_data_getitem(model_data: LanguageModelData, test_data_size: int) -> None:
    index_info_matrix = model_data.info_matrix.index
    assert len(index_info_matrix) == test_data_size - 1

    index_cosine_similarity_ranks = model_data.cosine_similarity_ranks.index
    # TODO: Why is this number not 99 (i.e. test_data_size - 1)?
    assert len(index_cosine_similarity_ranks) == 999

    # index of info matrix is subset of cosine similarity ranks index
    shared_indices = index_info_matrix.intersection(index_cosine_similarity_ranks)
    assert len(shared_indices) == test_data_size - 1

    # check that slicing works for info matrix, cosine similarity ranks matrix and integer labels
    sliced_model_data = model_data[shared_indices]
    assert isinstance(sliced_model_data, LanguageModelData)

    assert len(sliced_model_data.info_matrix) == len(shared_indices)
    assert len(sliced_model_data.integer_labels) == len(shared_indices)
    assert len(sliced_model_data.cosine_similarity_ranks) == len(shared_indices)
