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
from readnext.utils.aliases import (
    CitationFeaturesFrame,
    CitationInfoFrame,
    CitationPointsFrame,
    CitationRanksFrame,
    IntegerLabelsFrame,
    LanguageFeaturesFrame,
    LanguageInfoFrame,
)

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
def test_model_data_integer_labels_frame(
    model_data_constructor_integer_labels_frame: IntegerLabelsFrame,
) -> None:
    assert isinstance(model_data_constructor_integer_labels_frame, pl.DataFrame)

    assert model_data_constructor_integer_labels_frame.shape[1] == 2
    assert model_data_constructor_integer_labels_frame.columns == [
        "candidate_d3_document_id",
        "integer_label",
    ]

    assert model_data_constructor_integer_labels_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert model_data_constructor_integer_labels_frame["integer_label"].dtype == pl.Int64

    # check that all integer labels are either 0 or 1
    assert all(
        label in [0, 1] for label in model_data_constructor_integer_labels_frame["integer_label"]
    )


@pytest.mark.updated
def test_citation_model_data_info_frame(
    citation_model_data_constructor_info_frame: CitationInfoFrame,
) -> None:
    assert isinstance(citation_model_data_constructor_info_frame, pl.DataFrame)

    assert citation_model_data_constructor_info_frame.shape[1] == 6
    assert citation_model_data_constructor_info_frame.columns == [
        "candidate_d3_document_id",
        "title",
        "author",
        "arxiv_labels",
        "semanticscholar_url",
        "arxiv_url",
    ]

    assert citation_model_data_constructor_info_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert citation_model_data_constructor_info_frame["title"].dtype == pl.Utf8
    assert citation_model_data_constructor_info_frame["author"].dtype == pl.Utf8
    assert citation_model_data_constructor_info_frame["arxiv_labels"].dtype == pl.List(pl.Utf8)
    assert citation_model_data_constructor_info_frame["semanticscholar_url"].dtype == pl.Utf8
    assert citation_model_data_constructor_info_frame["arxiv_url"].dtype == pl.Utf8


@pytest.mark.updated
def test_language_model_data_info_frame(
    language_model_data_constructor_info_frame: LanguageInfoFrame,
) -> None:
    assert isinstance(language_model_data_constructor_info_frame, pl.DataFrame)

    assert language_model_data_constructor_info_frame.shape[1] == 6
    assert language_model_data_constructor_info_frame.columns == [
        "candidate_d3_document_id",
        "title",
        "author",
        "publication_date",
        "arxiv_labels",
        "semanticscholar_url",
        "arxiv_url",
    ]

    assert language_model_data_constructor_info_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert language_model_data_constructor_info_frame["title"].dtype == pl.Utf8
    assert language_model_data_constructor_info_frame["author"].dtype == pl.Utf8
    assert language_model_data_constructor_info_frame["arxiv_labels"].dtype == pl.List(pl.Utf8)
    assert language_model_data_constructor_info_frame["publication_date"].dtype == pl.Utf8
    assert language_model_data_constructor_info_frame["semanticscholar_url"].dtype == pl.Utf8
    assert language_model_data_constructor_info_frame["arxiv_url"].dtype == pl.Utf8


@pytest.mark.updated
def test_citation_model_data_features_frame(
    citation_model_data_constructor_features_frame: CitationFeaturesFrame,
) -> None:
    assert isinstance(citation_model_data_constructor_features_frame, pl.DataFrame)

    assert citation_model_data_constructor_features_frame.shape[1] == 6
    assert citation_model_data_constructor_features_frame.columns == [
        "candidate_d3_document_id",
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis_score",
        "bibliographic_coupling_score",
    ]

    assert (
        citation_model_data_constructor_features_frame["candidate_d3_document_id"].dtype == pl.Int64
    )
    assert citation_model_data_constructor_features_frame["publication_date"].dtype == pl.Utf8
    assert (
        citation_model_data_constructor_features_frame["citationcount_document"].dtype == pl.Int64
    )
    assert citation_model_data_constructor_features_frame["citationcount_author"].dtype == pl.Int64
    assert (
        citation_model_data_constructor_features_frame["co_citation_analysis_score"].dtype
        == pl.Float64
    )
    assert (
        citation_model_data_constructor_features_frame["bibliographic_coupling_score"].dtype
        == pl.Float64
    )


@pytest.mark.updated
def test_language_model_data_features_frame(
    language_model_data_constructor_features_frame: LanguageFeaturesFrame,
) -> None:
    assert isinstance(language_model_data_constructor_features_frame, pl.DataFrame)

    assert language_model_data_constructor_features_frame.shape[1] == 2
    assert language_model_data_constructor_features_frame.columns == [
        "candidate_d3_document_id",
        "cosine_similarity",
    ]

    assert (
        language_model_data_constructor_features_frame["candidate_d3_document_id"].dtype == pl.Int64
    )
    assert language_model_data_constructor_features_frame["cosine_similarity"].dtype == pl.Float64


@pytest.mark.updated
def test_citation_model_data_ranks_frame(
    citation_model_data_constructor_ranks_frame: CitationRanksFrame,
) -> None:
    assert isinstance(citation_model_data_constructor_ranks_frame, pl.DataFrame)

    assert citation_model_data_constructor_ranks_frame.shape[1] == 5
    assert citation_model_data_constructor_ranks_frame.columns == [
        "candidate_d3_document_id",
        "publication_date_rank",
        "citationcount_document_rank",
        "citationcount_author_rank",
        "co_citation_analysis_rank",
        "bibliographic_coupling_rank",
    ]

    assert (
        citation_model_data_constructor_ranks_frame["candidate_d3_document_id"].dtype == pl.Float32
    )
    assert citation_model_data_constructor_ranks_frame["publication_date_rank"].dtype == pl.Float32
    assert (
        citation_model_data_constructor_ranks_frame["citationcount_document_rank"].dtype
        == pl.Float32
    )
    assert (
        citation_model_data_constructor_ranks_frame["citationcount_author_rank"].dtype == pl.Float32
    )
    assert (
        citation_model_data_constructor_ranks_frame["co_citation_analysis_rank"].dtype == pl.Float32
    )
    assert (
        citation_model_data_constructor_ranks_frame["bibliographic_coupling_rank"].dtype
        == pl.Float32
    )

    # check that all ranks are between 1 and 101
    rank_columns = citation_model_data_constructor_ranks_frame.select(
        pl.exclude("candidate_d3_document_id")
    )

    column_minimums = rank_columns.min()
    assert (column_minimums.min(axis=1) == 1.0).all()
    assert (column_minimums.max(axis=1) == 1.0).all()

    column_maximums = rank_columns.max()
    assert (column_maximums.max(axis=1) == 101.0).all()
    assert (column_maximums.min(axis=1) == 101.0).all()


@pytest.mark.updated
def test_citation_model_data_points_frame(
    citation_model_data_constructor_points_frame: CitationPointsFrame,
) -> None:
    assert isinstance(citation_model_data_constructor_points_frame, pl.DataFrame)

    assert citation_model_data_constructor_points_frame.shape[1] == 5
    assert citation_model_data_constructor_points_frame.columns == [
        "candidate_d3_document_id",
        "publication_date_points",
        "citationcount_document_points",
        "citationcount_author_points",
        "co_citation_analysis_points",
        "bibliographic_coupling_points",
    ]

    assert (
        citation_model_data_constructor_points_frame["candidate_d3_document_id"].dtype == pl.Float32
    )
    assert (
        citation_model_data_constructor_points_frame["publication_date_points"].dtype == pl.Float32
    )
    assert (
        citation_model_data_constructor_points_frame["citationcount_document_points"].dtype
        == pl.Float32
    )
    assert (
        citation_model_data_constructor_points_frame["citationcount_author_points"].dtype
        == pl.Float32
    )
    assert (
        citation_model_data_constructor_points_frame["co_citation_analysis_points"].dtype
        == pl.Float32
    )
    assert (
        citation_model_data_constructor_points_frame["bibliographic_coupling_points"].dtype
        == pl.Float32
    )

    # check that all points are between 0 and 100
    rank_columns = citation_model_data_constructor_points_frame.select(
        pl.exclude("candidate_d3_document_id")
    )

    column_minimums = rank_columns.min()
    assert (column_minimums.min(axis=1) == 0.0).all()
    assert (column_minimums.max(axis=1) == 0.0).all()

    column_maximums = rank_columns.max()
    assert (column_maximums.max(axis=1) == 100.0).all()
    assert (column_maximums.min(axis=1) == 100.0).all()


@pytest.mark.updated
def test_kw_only_initialization_citation_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        CitationModelDataConstructor(
            -1,  # type: ignore
            pl.DataFrame(),
            SeenModelDataConstructorPlugin(d3_document_id=1, documents_frame=pl.DataFrame()),
            pl.DataFrame(),
            pl.DataFrame(),
        )


@pytest.mark.updated
def test_kw_only_initialization_language_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        LanguageModelDataConstructor(
            -1,  # type: ignore
            SeenModelDataConstructorPlugin(d3_document_id=1, documents_frame=pl.DataFrame()),
            pl.DataFrame(),
            pl.DataFrame(),
        )
