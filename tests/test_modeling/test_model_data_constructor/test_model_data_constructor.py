import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import DocumentInfo, ModelDataConstructor

citation_model_data_constructor_fixtures = ["citation_model_data_constructor"]
language_model_data_constructor_fixtures = ["language_model_data_constructor"]
model_data_constructor_fixtures = (
    citation_model_data_constructor_fixtures + language_model_data_constructor_fixtures
)


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_initialization(model_data_constructor: ModelDataConstructor) -> None:
    assert isinstance(model_data_constructor, ModelDataConstructor)

    assert isinstance(model_data_constructor.d3_document_id, int)
    assert model_data_constructor.d3_document_id == 13756489

    # number of columns is different betwen citation and language model data and tested
    # in individual tests below
    assert isinstance(model_data_constructor.documents_data, pl.DataFrame)

    assert isinstance(model_data_constructor.info_columns, list)
    assert all(isinstance(col, str) for col in model_data_constructor.info_columns)

    assert isinstance(model_data_constructor.query_document, DocumentInfo)


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_exclude_query_document(
    model_data_constructor: ModelDataConstructor,
) -> None:
    excluded_df = model_data_constructor.exclude_query_document(
        model_data_constructor.documents_data
    )

    assert isinstance(excluded_df, pl.DataFrame)
    assert model_data_constructor.d3_document_id not in excluded_df.index


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_filter_documents_data(
    model_data_constructor: ModelDataConstructor,
) -> None:
    filtered_df = model_data_constructor.filter_documents_data()

    assert isinstance(filtered_df, pl.DataFrame)
    assert model_data_constructor.d3_document_id not in filtered_df.index


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_get_info_matrix(model_data_constructor: ModelDataConstructor) -> None:
    info_matrix = model_data_constructor.get_info_frame()

    assert isinstance(info_matrix, pl.DataFrame)
    assert model_data_constructor.d3_document_id not in info_matrix.index
    assert all(col in info_matrix.columns for col in model_data_constructor.info_columns)


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_shares_arxiv_label(model_data_constructor: ModelDataConstructor) -> None:
    candidate_document_labels = ["cs.CL", "stat.ML"]
    result = model_data_constructor.shares_arxiv_label(candidate_document_labels)

    assert isinstance(result, bool)
    assert result

    candidate_document_labels = ["cs.CV"]
    result = model_data_constructor.shares_arxiv_label(candidate_document_labels)

    assert isinstance(result, bool)
    assert not result


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_boolean_to_int(model_data_constructor: ModelDataConstructor) -> None:
    result = model_data_constructor.boolean_to_int(True)

    assert isinstance(result, int)
    assert result == 1
