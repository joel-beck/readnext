import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.data.semanticscholar import SemanticScholarResponse
from readnext.inference.attribute_getter import (
    QueryCitationModelDataConstructor,
    QueryLanguageModelDataConstructor,
)
from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    DocumentScore,
    LanguageModelDataConstructor,
    ModelDataConstructor,
)

citation_model_data_constructor_fixtures = ["citation_model_data_constructor"]
language_model_data_constructor_fixtures = ["language_model_data_constructor"]
model_data_constructor_fixtures = (
    citation_model_data_constructor_fixtures + language_model_data_constructor_fixtures
)


# SECTION: ModelDataConstructor
@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_initialization(model_data_constructor: ModelDataConstructor) -> None:
    assert isinstance(model_data_constructor, ModelDataConstructor)

    assert isinstance(model_data_constructor.d3_document_id, int)
    assert model_data_constructor.d3_document_id == 206594692

    # number of columns is different betwen citation and language model data and tested
    # in individual tests below
    assert isinstance(model_data_constructor.documents_data, pd.DataFrame)

    assert isinstance(model_data_constructor.info_cols, list)
    assert all(isinstance(col, str) for col in model_data_constructor.info_cols)

    assert isinstance(model_data_constructor.query_document, DocumentInfo)


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_collect_query_document(
    model_data_constructor: ModelDataConstructor,
) -> None:
    assert isinstance(model_data_constructor.query_document.d3_document_id, int)
    assert model_data_constructor.query_document.d3_document_id == 206594692

    assert isinstance(model_data_constructor.query_document.title, str)
    assert (
        model_data_constructor.query_document.title
        == "Deep Residual Learning for Image Recognition"
    )

    assert isinstance(model_data_constructor.query_document.author, str)
    assert model_data_constructor.query_document.author == "Kaiming He"

    assert isinstance(model_data_constructor.query_document.arxiv_labels, list)
    assert model_data_constructor.query_document.arxiv_labels == ["cs.CV"]


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

    assert isinstance(excluded_df, pd.DataFrame)
    assert model_data_constructor.d3_document_id not in excluded_df.index


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_filter_documents_data(
    model_data_constructor: ModelDataConstructor,
) -> None:
    filtered_df = model_data_constructor.filter_documents_data()

    assert isinstance(filtered_df, pd.DataFrame)
    assert model_data_constructor.d3_document_id not in filtered_df.index


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_get_info_matrix(model_data_constructor: ModelDataConstructor) -> None:
    info_matrix = model_data_constructor.get_info_matrix()

    assert isinstance(info_matrix, pd.DataFrame)
    assert model_data_constructor.d3_document_id not in info_matrix.index
    assert all(col in info_matrix.columns for col in model_data_constructor.info_cols)


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_extend_info_matrix(model_data_constructor: ModelDataConstructor) -> None:
    info_matrix = model_data_constructor.get_info_matrix()
    extended_matrix = model_data_constructor.extend_info_matrix(info_matrix)

    assert isinstance(extended_matrix, pd.DataFrame)
    assert all(col in extended_matrix.columns for col in model_data_constructor.info_cols)


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_shares_arxiv_label(model_data_constructor: ModelDataConstructor) -> None:
    candidate_document_labels = ["cs.CV", "stat.ML"]
    result = model_data_constructor.shares_arxiv_label(candidate_document_labels)

    assert isinstance(result, bool)
    assert result is True

    candidate_document_labels = ["cs.AI", "cs.LG"]
    result = model_data_constructor.shares_arxiv_label(candidate_document_labels)

    assert isinstance(result, bool)
    assert result is False


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_boolean_to_int(model_data_constructor: ModelDataConstructor) -> None:
    result = model_data_constructor.boolean_to_int(True)

    assert isinstance(result, int)
    assert result == 1


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_get_integer_labels(model_data_constructor: ModelDataConstructor) -> None:
    integer_labels = model_data_constructor.get_integer_labels()

    assert isinstance(integer_labels, pd.Series)


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(model_data_constructor_fixtures),
)
def test_document_scores_to_frame(model_data_constructor: ModelDataConstructor) -> None:
    document_scores = [
        DocumentScore(
            document_info=DocumentInfo(
                d3_document_id=1, title="A", author="A.A", arxiv_labels=["cs.CV"]
            ),
            score=0.5,
        ),
        DocumentScore(
            document_info=DocumentInfo(
                d3_document_id=2, title="B", author="B.B", arxiv_labels=["stat.ML"]
            ),
            score=0.3,
        ),
    ]
    scores_df = model_data_constructor.document_scores_to_frame(document_scores)

    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape[1] == 1
    assert scores_df.columns.to_list() == ["score"]
    assert scores_df.index.name == "document_id"


# SECTION: CitationModelDataConstructor
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
    scores_df = model_data_constructor.get_citation_method_scores(citation_method_data)

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
    "model_data_constructor",
    lazy_fixture(citation_model_data_constructor_fixtures),
)
def test_get_feature_matrix(
    model_data_constructor: CitationModelDataConstructor,
) -> None:
    feature_matrix = model_data_constructor.get_feature_matrix()

    assert isinstance(feature_matrix, pd.DataFrame)
    assert feature_matrix.shape[1] == len(model_data_constructor.feature_cols) + 2
    assert "co_citation_analysis_rank" in feature_matrix.columns
    assert "bibliographic_coupling_rank" in feature_matrix.columns


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


# SECTION: LanguageModelDataConstructor
@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(language_model_data_constructor_fixtures),
)
def test_language_model_constructor_initialization(
    model_data_constructor: LanguageModelDataConstructor,
) -> None:
    assert model_data_constructor.info_cols == ["title", "author", "arxiv_labels"]

    assert isinstance(model_data_constructor.cosine_similarities, pd.DataFrame)
    assert model_data_constructor.cosine_similarities.shape[1] == 1

    assert model_data_constructor.documents_data.shape[1] == 24


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(language_model_data_constructor_fixtures),
)
def test_get_cosine_similarity_scores(
    model_data_constructor: LanguageModelDataConstructor,
) -> None:
    scores_df = model_data_constructor.get_cosine_similarity_scores()

    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape[1] == 1
    assert scores_df.columns.to_list() == ["cosine_similarity"]
    assert scores_df.index.name == "document_id"


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(language_model_data_constructor_fixtures),
)
def test_extend_info_matrix_language_model(
    model_data_constructor: LanguageModelDataConstructor,
) -> None:
    info_matrix = model_data_constructor.get_info_matrix()
    extended_matrix = model_data_constructor.extend_info_matrix(info_matrix)

    assert isinstance(extended_matrix, pd.DataFrame)
    assert extended_matrix.shape[1] == len(model_data_constructor.info_cols) + 1
    assert "cosine_similarity" in extended_matrix.columns


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(language_model_data_constructor_fixtures),
)
def test_get_cosine_similarity_ranks(
    model_data_constructor: LanguageModelDataConstructor,
) -> None:
    ranks_df = model_data_constructor.get_cosine_similarity_ranks()

    assert isinstance(ranks_df, pd.DataFrame)
    assert ranks_df.shape[1] == 1
    assert "cosine_similarity_rank" in ranks_df.columns
    assert ranks_df.index.name == "document_id"


def test_kw_only_initialization_language_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        LanguageModelDataConstructor(
            -1,  # type: ignore
            pd.DataFrame(),
            pd.DataFrame(),
        )


def test_kw_only_initialization_query_language_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        QueryLanguageModelDataConstructor(
            -1,  # type: ignore
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
