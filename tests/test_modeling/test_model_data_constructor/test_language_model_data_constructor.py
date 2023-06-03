import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.data.semanticscholar import SemanticScholarResponse
from readnext.inference.attribute_getter import QueryLanguageModelDataConstructor
from readnext.modeling import LanguageModelData, LanguageModelDataConstructor

language_model_data_constructor_fixtures = ["language_model_data_constructor"]


@pytest.mark.parametrize(
    "model_data_constructor",
    lazy_fixture(language_model_data_constructor_fixtures),
)
def test_language_model_constructor_initialization(
    model_data_constructor: LanguageModelDataConstructor,
) -> None:
    assert model_data_constructor.info_cols == ["title", "author", "arxiv_labels"]

    assert isinstance(model_data_constructor.cosine_similarities, pl.DataFrame)
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

    assert isinstance(scores_df, pl.DataFrame)
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

    assert isinstance(extended_matrix, pl.DataFrame)
    assert extended_matrix.shape[1] == len(model_data_constructor.info_cols) + 1
    assert "cosine_similarity" in extended_matrix.columns


@pytest.mark.parametrize(
    "model_data_constructor", lazy_fixture(language_model_data_constructor_fixtures)
)
def test_language_model_data_from_constructor(
    model_data_constructor: LanguageModelDataConstructor,
) -> None:
    language_model_data = LanguageModelData.from_constructor(model_data_constructor)

    assert isinstance(language_model_data, LanguageModelData)
    assert isinstance(language_model_data.info_matrix, pl.DataFrame)
    assert isinstance(language_model_data.integer_labels, pl.Series)
    assert isinstance(language_model_data.cosine_similarity_ranks, pl.DataFrame)


def test_kw_only_initialization_language_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        LanguageModelDataConstructor(
            -1,  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
        )


def test_kw_only_initialization_query_language_model_data_constructor() -> None:
    with pytest.raises(TypeError):
        QueryLanguageModelDataConstructor(
            -1,  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            SemanticScholarResponse(
                semanticscholar_id="SemantischscholarID",
                arxiv_id="ArxviID",
                title="Title",
                abstract="Abstract",
                citations=[],
                references=[],
            ),
        )
