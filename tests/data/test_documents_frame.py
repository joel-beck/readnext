import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import DocumentsFrame

documents_frame_fixtures = [lazy_fixture("test_documents_frame")]
documents_frame_fixtures_slow_skip_ci = [lazy_fixture("inference_data_constructor_documents_frame")]


@pytest.mark.parametrize(
    "documents_frame",
    [
        *[pytest.param(fixture) for fixture in documents_frame_fixtures],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in documents_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_column_names(documents_frame: DocumentsFrame) -> None:
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

    assert documents_frame.columns == columns


@pytest.mark.parametrize(
    "documents_frame",
    [
        *[pytest.param(fixture) for fixture in documents_frame_fixtures],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in documents_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_dtypes(documents_frame: DocumentsFrame) -> None:
    assert documents_frame["d3_document_id"].dtype == pl.Int64
    assert documents_frame["d3_author_id"].dtype == pl.Int64
    assert documents_frame["title"].dtype == pl.Utf8
    assert documents_frame["author"].dtype == pl.Utf8
    assert documents_frame["publication_date"].dtype == pl.Utf8
    assert documents_frame["publication_date_rank"].dtype == pl.Float32
    assert documents_frame["citationcount_document"].dtype == pl.Int64
    assert documents_frame["citationcount_document_rank"].dtype == pl.Float32
    assert documents_frame["citationcount_author"].dtype == pl.Int64
    assert documents_frame["citationcount_author_rank"].dtype == pl.Float32
    assert documents_frame["citations"].dtype == pl.List
    assert documents_frame["references"].dtype == pl.List
    assert documents_frame["abstract"].dtype == pl.Utf8
    assert documents_frame["semanticscholar_id"].dtype == pl.Utf8
    assert documents_frame["semanticscholar_url"].dtype == pl.Utf8
    assert documents_frame["semanticscholar_tags"].dtype == pl.List
    assert documents_frame["arxiv_id"].dtype == pl.Utf8
    assert documents_frame["arxiv_url"].dtype == pl.Utf8
    assert documents_frame["arxiv_labels"].dtype == pl.List


@pytest.mark.parametrize(
    "documents_frame",
    [
        *[pytest.param(fixture) for fixture in documents_frame_fixtures],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in documents_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_arxiv_labels(documents_frame: DocumentsFrame) -> None:
    arxiv_labels = documents_frame["arxiv_labels"]
    first_observation = arxiv_labels[0].to_list()

    assert isinstance(first_observation, list)
    assert isinstance(first_observation[0], str)

    # Check that all observations have at least one label
    assert arxiv_labels.apply(lambda x: len(x) > 0).all()

    unique_arxiv_labels = {label for labels in arxiv_labels for label in labels}
    assert len(unique_arxiv_labels) == 16


@pytest.mark.parametrize(
    "documents_frame",
    [
        *[pytest.param(fixture) for fixture in documents_frame_fixtures],
        *[
            pytest.param(fixture, marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in documents_frame_fixtures_slow_skip_ci
        ],
    ],
)
def test_semanticscholar_tags(documents_frame: DocumentsFrame) -> None:
    semanticscholar_tags = documents_frame["semanticscholar_tags"]
    first_observation = semanticscholar_tags[0].to_list()

    assert isinstance(first_observation, list)
    assert isinstance(first_observation[0], str)

    # Check that all observations have at least one tag
    assert semanticscholar_tags.apply(lambda x: len(x) > 0).all()

    unique_semanticscholar_tags = {tag for tags in semanticscholar_tags for tag in tags}
    assert len(unique_semanticscholar_tags) == 7
