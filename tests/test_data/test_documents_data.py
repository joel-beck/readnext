import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype, is_string_dtype
from pytest_lazyfixture import lazy_fixture

full_documents_data_fixtures = [
    "documents_authors_labels_citations_most_cited",
]

subset_documents_data_fixtures = [
    "inference_data_seen_constructor_documents_data",
    "inference_data_unseen_constructor_documents_data",
]

documents_data_fixtures = full_documents_data_fixtures + subset_documents_data_fixtures


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(documents_data_fixtures))
def test_index(
    documents_data: pd.DataFrame,
) -> None:
    assert documents_data.index.name == "document_id"
    assert documents_data.index.is_unique
    assert documents_data.index.dtype == pd.Int64Dtype()


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(documents_data_fixtures))
def test_contains_columns_subset(
    documents_data: pd.DataFrame,
) -> None:
    columns_subset = [
        "author_id",
        "title",
        "author",
        "publication_date",
        "publication_year",
        "citationcount_document",
        "citationcount_author",
        "abstract",
        "arxiv_id",
        "arxiv_labels",
        "semanticscholar_url",
        "semanticscholar_tags",
    ]

    assert set(columns_subset).issubset(set(documents_data.columns))


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(documents_data_fixtures))
def test_dtypes(
    documents_data: pd.DataFrame,
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


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(documents_data_fixtures))
def test_arxiv_labels(
    documents_data: pd.DataFrame,
) -> None:
    arxiv_labels = documents_data["arxiv_labels"]
    first_observation = arxiv_labels.iloc[0]

    assert isinstance(first_observation, list)
    assert isinstance(first_observation[0], str)

    # Check that all observations have at least one label
    assert arxiv_labels.apply(lambda x: len(x) > 0).all()


@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(full_documents_data_fixtures))
def test_arxiv_labels_full_documents_data(
    documents_data: pd.DataFrame,
) -> None:
    arxiv_labels = documents_data["arxiv_labels"]
    # `col.sum()` for a dataframe column containing lists returns a set of all unique
    # values!
    unique_arxiv_labels = set(arxiv_labels.sum())

    # Check that all 40 arxiv labels within computer science are contained in the
    # dataset
    assert len(unique_arxiv_labels) == 40


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(subset_documents_data_fixtures))
def test_arxiv_labels_subset_documents_data(
    documents_data: pd.DataFrame,
) -> None:
    arxiv_labels = documents_data["arxiv_labels"]
    # `col.sum()` for a dataframe column containing lists returns a set of all unique
    # values!
    unique_arxiv_labels = set(arxiv_labels.sum())

    # Check that all 40 arxiv labels within computer science are contained in the
    # dataset
    assert len(unique_arxiv_labels) == 35


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(documents_data_fixtures))
def test_semanticscholar_tags(
    documents_data: pd.DataFrame,
) -> None:
    semanticscholar_tags = documents_data["semanticscholar_tags"]
    first_observation = semanticscholar_tags.iloc[0]

    assert isinstance(first_observation, list)
    assert isinstance(first_observation[0], str)

    # Check that all observations have at least one tag
    assert semanticscholar_tags.apply(lambda x: len(x) > 0).all()


@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(full_documents_data_fixtures))
def test_semanticscholar_tags_full_documents_data(
    documents_data: pd.DataFrame,
) -> None:
    semanticscholar_tags = documents_data["semanticscholar_tags"]

    ## 22 unique tags in total
    unique_semanticscholar_tags = set(semanticscholar_tags.sum())
    assert len(unique_semanticscholar_tags) == 22


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("documents_data", lazy_fixture(subset_documents_data_fixtures))
def test_semanticscholar_tags_subset_documents_data(
    documents_data: pd.DataFrame,
) -> None:
    semanticscholar_tags = documents_data["semanticscholar_tags"]

    ## 22 unique tags in total
    unique_semanticscholar_tags = set(semanticscholar_tags.sum())
    assert len(unique_semanticscholar_tags) == 19
