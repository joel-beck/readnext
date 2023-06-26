import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelDataConstructor,
    ModelDataConstructor,
    ModelDataConstructorPlugin,
)
from readnext.modeling.constructor_plugin import SeenModelDataConstructorPlugin

citation_model_data_constructor_fixtures = [
    "citation_model_data_constructor_seen",
    "citation_model_data_constructor_unseen",
]

language_model_data_constructor_fixtures = [
    "language_model_data_constructor_seen",
    "language_model_data_constructor_unseen",
]

model_data_constructor_fixtures = (
    citation_model_data_constructor_fixtures + language_model_data_constructor_fixtures
)


@pytest.mark.updated
@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_initialization(model_data_constructor: ModelDataConstructor) -> None:
    assert isinstance(model_data_constructor, ModelDataConstructor)

    assert isinstance(model_data_constructor.d3_document_id, int)
    assert isinstance(model_data_constructor.documents_frame, pl.DataFrame)
    assert isinstance(model_data_constructor.constructor_plugin, ModelDataConstructorPlugin)

    assert isinstance(model_data_constructor.info_columns, list)
    assert all(isinstance(col, str) for col in model_data_constructor.info_columns)

    assert isinstance(model_data_constructor.feature_columns, list)
    assert all(isinstance(col, str) for col in model_data_constructor.feature_columns)

    assert isinstance(model_data_constructor.query_document, DocumentInfo)


@pytest.mark.updated
@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_get_query_documents_frame(
    model_data_constructor: ModelDataConstructor,
) -> None:
    query_documents_frame = model_data_constructor.get_query_documents_frame()

    assert isinstance(query_documents_frame, pl.DataFrame)

    # check that id column is renamed
    assert "d3_document_id" not in query_documents_frame.columns
    assert "candidate_d3_document_id" in query_documents_frame.columns

    # check that query id is filtered out
    assert model_data_constructor.d3_document_id not in query_documents_frame["d3_document_id"]


@pytest.mark.updated
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


@pytest.mark.updated
@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_boolean_to_int(model_data_constructor: ModelDataConstructor) -> None:
    result = model_data_constructor.boolean_to_int(True)

    assert isinstance(result, int)
    assert result == 1


@pytest.mark.updated
def test_kw_only_initialization_citation_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        CitationModelDataConstructor(
            -1,  # type: ignore
            pl.DataFrame(),
            SeenModelDataConstructorPlugin(d3_document_id=1, documents_frame=pl.DataFrame()),
            co_citation_analysis_scores_frame=pl.DataFrame(),
            bibliographic_coupling_scores_frame=pl.DataFrame(),
        )


@pytest.mark.updated
def test_kw_only_initialization_language_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        LanguageModelDataConstructor(
            -1,  # type: ignore
            pl.DataFrame(),
            SeenModelDataConstructorPlugin(d3_document_id=1, documents_frame=pl.DataFrame()),
            cosine_similarity_scores_frame=pl.DataFrame(),
        )
