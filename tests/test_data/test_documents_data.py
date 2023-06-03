import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

documents_data_fixtures_skip_ci = ["documents_data"]

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
    assert documents_data["d3_document_id"].dtype == pl.Int64
    assert documents_data["d3_author_id"].dtype == pl.Int64
    assert documents_data["title"].dtype == pl.Utf8
    assert documents_data["author"].dtype == pl.Utf8
    assert documents_data["publication_date"].dtype == pl.Utf8
    assert documents_data["publication_date_rank"].dtype == pl.Int64
    assert documents_data["citationcount_document"].dtype == pl.Int64
    assert documents_data["citationcount_document_rank"].dtype == pl.Int64
    assert documents_data["citationcount_author"].dtype == pl.Int64
    assert documents_data["citationcount_author_rank"].dtype == pl.Int64
    assert documents_data["citations"].dtype == pl.List
    assert documents_data["references"].dtype == pl.List
    assert documents_data["abstract"].dtype == pl.Utf8
    assert documents_data["semanticscholar_id"].dtype == pl.Utf8
    assert documents_data["semanticscholar_url"].dtype == pl.Utf8
    assert documents_data["semanticscholar_tags"].dtype == pl.List
    assert documents_data["arxiv_id"].dtype == pl.Utf8
    assert documents_data["arxiv_url"].dtype == pl.Utf8


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
    first_observation = arxiv_labels[0]

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
    unique_arxiv_labels = {label for labels in arxiv_labels for label in labels}

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
    unique_arxiv_labels = {label for labels in arxiv_labels for label in labels}

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
    first_observation = semanticscholar_tags[0]

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

    unique_semanticscholar_tags = {tag for tags in semanticscholar_tags for tag in tags}
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
    unique_semanticscholar_tags = {tag for tags in semanticscholar_tags for tag in tags}
    assert len(unique_semanticscholar_tags) == 19
