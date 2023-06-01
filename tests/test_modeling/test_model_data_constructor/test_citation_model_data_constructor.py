import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.data.semanticscholar import SemanticScholarResponse
from readnext.inference.attribute_getter import QueryCitationModelDataConstructor
from readnext.modeling import CitationModelData, CitationModelDataConstructor

citation_model_data_constructor_fixtures = ["citation_model_data_constructor"]


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(citation_model_data_constructor_fixtures),
)
def test_citation_model_constructor_initialization(
    model_data_constructor: CitationModelDataConstructor,
) -> None:
    assert model_data_constructor.info_cols == [
        "title",
        "author",
        "arxiv_labels",
        "publication_date",
        "citationcount_document",
        "citationcount_author",
    ]
    assert model_data_constructor.feature_cols == [
        "publication_date_rank",
        "citationcount_document_rank",
        "citationcount_author_rank",
    ]

    assert isinstance(model_data_constructor.co_citation_analysis_scores, pd.DataFrame)
    assert model_data_constructor.co_citation_analysis_scores.shape[1] == 1

    assert isinstance(model_data_constructor.bibliographic_coupling_scores, pd.DataFrame)
    assert model_data_constructor.bibliographic_coupling_scores.shape[1] == 1

    assert model_data_constructor.documents_data.shape[1] == 25


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(citation_model_data_constructor_fixtures),
)
def test_get_citation_method_scores(
    model_data_constructor: CitationModelDataConstructor,
) -> None:
    citation_method_data = model_data_constructor.co_citation_analysis_scores
    scores_df = model_data_constructor.get_query_scores(citation_method_data)

    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape[1] == 1
    assert scores_df.columns.to_list() == ["score"]
    assert scores_df.index.name == "document_id"


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(citation_model_data_constructor_fixtures),
)
def test_get_co_citation_analysis_scores(
    model_data_constructor: CitationModelDataConstructor,
) -> None:
    co_citation_analysis_scores = model_data_constructor.get_co_citation_analysis_scores()

    assert isinstance(co_citation_analysis_scores, pd.DataFrame)
    assert co_citation_analysis_scores.shape[1] == 1
    assert co_citation_analysis_scores.columns.to_list() == ["co_citation_analysis"]
    assert co_citation_analysis_scores.index.name == "document_id"


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(citation_model_data_constructor_fixtures),
)
def test_get_bibliographic_coupling_scores(
    model_data_constructor: CitationModelDataConstructor,
) -> None:
    bibliographic_coupling_scores = model_data_constructor.get_bibliographic_coupling_scores()

    assert isinstance(bibliographic_coupling_scores, pd.DataFrame)
    assert bibliographic_coupling_scores.shape[1] == 1
    assert bibliographic_coupling_scores.columns.to_list() == ["bibliographic_coupling"]
    assert bibliographic_coupling_scores.index.name == "document_id"


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(citation_model_data_constructor_fixtures),
)
def test_extend_info_matrix_citation_model(
    model_data_constructor: CitationModelDataConstructor,
) -> None:
    info_matrix = model_data_constructor.get_info_matrix()
    extended_matrix = model_data_constructor.extend_info_matrix(info_matrix)

    assert isinstance(extended_matrix, pd.DataFrame)
    assert extended_matrix.shape[1] == len(model_data_constructor.info_cols) + 2
    assert "co_citation_analysis" in extended_matrix.columns
    assert "bibliographic_coupling" in extended_matrix.columns


@pytest.mark.parametrize(
    "model_data_constructor", lazy_fixture(citation_model_data_constructor_fixtures)
)
def test_citation_model_data_from_constructor(
    model_data_constructor: CitationModelDataConstructor,
) -> None:
    citation_model_data = CitationModelData.from_constructor(model_data_constructor)

    assert isinstance(citation_model_data, CitationModelData)
    assert isinstance(citation_model_data.info_matrix, pd.DataFrame)
    assert isinstance(citation_model_data.integer_labels, pd.Series)
    assert isinstance(citation_model_data.feature_matrix, pd.DataFrame)


def test_kw_only_initialization_citation_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        CitationModelDataConstructor(
            -1,  # type: ignore
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )


def test_kw_only_initialization_query_citation_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        QueryCitationModelDataConstructor(
            -1,  # type: ignore
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            SemanticScholarResponse(
                semanticscholar_id="SemantischscholarID",
                arxiv_id="ArxviID",
                title="Title",
                abstract="Abstract",
                citations=[],
                references=[],
            ),
        )
