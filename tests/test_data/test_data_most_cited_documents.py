import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype, is_string_dtype
from pandas.testing import assert_frame_equal

from readnext.config import DataPaths
from readnext.utils import load_df_from_pickle


@pytest.fixture(scope="module")
def documents_authors_labels_citations_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(DataPaths.merged.documents_authors_labels_citations_most_cited_pkl)


def test_contains_columns_subset(
    documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    columns_subset = [
        "document_id",
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

    assert set(columns_subset).issubset(set(documents_authors_labels_citations_most_cited.columns))


def test_dtypes(
    documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    is_integer_dtype(documents_authors_labels_citations_most_cited["document_id"])
    is_integer_dtype(documents_authors_labels_citations_most_cited["author_id"])
    is_string_dtype(documents_authors_labels_citations_most_cited["title"])
    is_string_dtype(documents_authors_labels_citations_most_cited["author"])
    is_string_dtype(documents_authors_labels_citations_most_cited["publication_date"])
    is_integer_dtype(documents_authors_labels_citations_most_cited["publication_year"])
    is_integer_dtype(documents_authors_labels_citations_most_cited["citationcount_document"])
    is_integer_dtype(documents_authors_labels_citations_most_cited["citationcount_author"])
    is_string_dtype(documents_authors_labels_citations_most_cited["abstract"])
    is_string_dtype(documents_authors_labels_citations_most_cited["arxiv_id"])
    is_string_dtype(documents_authors_labels_citations_most_cited["semanticscholar_url"])


def test_arxiv_labels(
    documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    arxiv_labels = documents_authors_labels_citations_most_cited["arxiv_labels"]
    first_observation = arxiv_labels.iloc[0]

    assert isinstance(first_observation, list)
    assert isinstance(first_observation[0], str)

    # Check that all observations have at least one label
    assert arxiv_labels.apply(lambda x: len(x) > 0).all()

    # `col.sum()` for a dataframe column containing lists returns a set of all unique
    # values!
    unique_arxiv_labels = set(arxiv_labels.sum())

    # Check that all 40 arxiv labels within computer science are contained in the
    # dataset
    assert len(unique_arxiv_labels) == 40


def test_semanticscholar_tags(
    documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    semanticscholar_tags = documents_authors_labels_citations_most_cited["semanticscholar_tags"]
    first_observation = semanticscholar_tags.iloc[0]

    assert isinstance(first_observation, list)
    assert isinstance(first_observation[0], str)

    # Check that all observations have at least one tag
    assert semanticscholar_tags.apply(lambda x: len(x) > 0).all()

    ## 22 unique tags in total
    unique_semanticscholar_tags = set(semanticscholar_tags.sum())
    assert len(unique_semanticscholar_tags) == 22


def test_that_test_data_mimics_real_data(
    test_data_size: int,
    documents_authors_labels_citations_most_cited: pd.DataFrame,
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    assert_frame_equal(
        documents_authors_labels_citations_most_cited.head(test_data_size),
        test_documents_authors_labels_citations_most_cited,
    )
