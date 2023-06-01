import polars as pl
import pytest
from pandas.api.types import is_integer_dtype, is_string_dtype
from pytest_lazyfixture import lazy_fixture

documents_data_fixtures_skip_ci = [
    "documents_authors_labels_citations_most_cited",
]

documents_data_fixtures_slow_skip_ci = [
    "inference_data_constructor_seen_documents_data",
    "inference_data_constructor_unseen_documents_data",
]


@pytest.mark.parametrize(
    "documents_data",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in documents_data_fixtures_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in documents_data_fixtures_slow_skip_ci
        ],
    ],
)
def test_column_names(
    documents_data: pl.DataFrame,
) -> None:
    columns = [
        "d3_document_id",
        "d3_author_id",
        "title",
        "author",
        "publication_date",
        "publication_date_rank",
        "citationcount_document",
        "citationcount_document_rank",
        "citationcount_author",
        "citationcount_author_rank",
        "citations",
        "references",
        "abstract",
        "semanticscholar_id",
        "semanticscholar_url",
        "semanticscholar_tags",
        "arxiv_id",
        "arxiv_url",
        "arxiv_labels",
    ]

    assert documents_data.columns == columns


@pytest.mark.parametrize(
    "documents_data",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in documents_data_fixtures_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in documents_data_fixtures_slow_skip_ci
        ],
    ],
)
def test_dtypes(
    documents_data: pl.DataFrame,
) -> None:
    is_integer_dtype(documents_data.index)
    is_integer_dtype(documents_data["author_id"])
    is_string_dtype(documents_data["title"])
    is_string_dtype(documents_data["author"])
    is_string_dtype(documents_data["publication_date"])
    is_integer_dtype(documents_data["publication_year"])
    is_integer_dtype(documents_data["citationcount_document"])
    is_integer_dtype(documents_data["citationcount_author"])
    is_string_dtype(documents_data["abstract"])
    is_string_dtype(documents_data["arxiv_id"])
    is_string_dtype(documents_data["semanticscholar_url"])


@pytest.mark.parametrize(
    "documents_data",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in documents_data_fixtures_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in documents_data_fixtures_slow_skip_ci
        ],
    ],
)
def test_arxiv_labels(
    documents_data: pl.DataFrame,
) -> None:
    arxiv_labels = documents_data["arxiv_labels"]
    first_observation = arxiv_labels.iloc[0]

    assert isinstance(first_observation, list)
    assert isinstance(first_observation[0], str)

    # Check that all observations have at least one label
    assert arxiv_labels.apply(lambda x: len(x) > 0).all()


@pytest.mark.parametrize(
    "documents_data",
    [
        pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
        for fixture in documents_data_fixtures_skip_ci
    ],
)
def test_arxiv_labels_full_documents_data(
    documents_data: pl.DataFrame,
) -> None:
    arxiv_labels = documents_data["arxiv_labels"]
    # `col.sum()` for a dataframe column containing lists returns a set of all unique
    # values!
    unique_arxiv_labels = set(arxiv_labels.sum())

    # Check that all 40 arxiv labels within computer science are contained in the
    # dataset
    assert len(unique_arxiv_labels) == 40


@pytest.mark.parametrize(
    "documents_data",
    [
        pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
        for fixture in documents_data_fixtures_slow_skip_ci
    ],
)
def test_arxiv_labels_subset_documents_data(
    documents_data: pl.DataFrame,
) -> None:
    arxiv_labels = documents_data["arxiv_labels"]
    # `col.sum()` for a dataframe column containing lists returns a set of all unique
    # values!
    unique_arxiv_labels = set(arxiv_labels.sum())

    # Check that all 40 arxiv labels within computer science are contained in the
    # dataset
    assert len(unique_arxiv_labels) == 35


@pytest.mark.parametrize(
    "documents_data",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in documents_data_fixtures_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in documents_data_fixtures_slow_skip_ci
        ],
    ],
)
def test_semanticscholar_tags(
    documents_data: pl.DataFrame,
) -> None:
    semanticscholar_tags = documents_data["semanticscholar_tags"]
    first_observation = semanticscholar_tags.iloc[0]

    assert isinstance(first_observation, list)
    assert isinstance(first_observation[0], str)

    # Check that all observations have at least one tag
    assert semanticscholar_tags.apply(lambda x: len(x) > 0).all()


@pytest.mark.parametrize(
    "documents_data",
    [
        pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
        for fixture in documents_data_fixtures_skip_ci
    ],
)
def test_semanticscholar_tags_full_documents_data(
    documents_data: pl.DataFrame,
) -> None:
    semanticscholar_tags = documents_data["semanticscholar_tags"]

    ## 22 unique tags in total
    unique_semanticscholar_tags = set(semanticscholar_tags.sum())
    assert len(unique_semanticscholar_tags) == 22


@pytest.mark.parametrize(
    "documents_data",
    [
        pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
        for fixture in documents_data_fixtures_slow_skip_ci
    ],
)
def test_semanticscholar_tags_subset_documents_data(
    documents_data: pl.DataFrame,
) -> None:
    semanticscholar_tags = documents_data["semanticscholar_tags"]

    ## 22 unique tags in total
    unique_semanticscholar_tags = set(semanticscholar_tags.sum())
    assert len(unique_semanticscholar_tags) == 19
