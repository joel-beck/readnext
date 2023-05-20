import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype


def test_index(
    documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    assert documents_authors_labels_citations_most_cited.index.name == "document_id"
    assert documents_authors_labels_citations_most_cited.index.is_unique
    assert documents_authors_labels_citations_most_cited.index.dtype == pd.Int64Dtype()


def test_contains_columns_subset(
    documents_authors_labels_citations_most_cited: pd.DataFrame,
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

    assert set(columns_subset).issubset(set(documents_authors_labels_citations_most_cited.columns))


def test_dtypes(
    documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    is_integer_dtype(documents_authors_labels_citations_most_cited.index)
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
