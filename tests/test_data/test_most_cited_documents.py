import pandas as pd
import pytest

from readnext.config import DataPaths


@pytest.fixture(scope="module")
def documents_authors_labels_citations_most_cited() -> pd.DataFrame:
    return pd.read_pickle(DataPaths.merged.documents_authors_labels_citations_most_cited_pkl)


def test_contains_important_columns(
    documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> None:
    important_columns = [
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

    assert set(important_columns).issubset(
        set(documents_authors_labels_citations_most_cited.columns)
    )
