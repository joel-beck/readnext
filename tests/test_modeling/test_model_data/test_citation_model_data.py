import numpy as np
import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import CitationModelData

# SUBSECTION: Test Getitem
citation_model_data_fixtures_seen = ["citation_model_data"]

citation_model_data_fixtures_seen_slow_skip_ci_test_data = [
    "seen_paper_attribute_getter_citation_model_data"
]
citation_model_data_fixtures_seen_slow_skip_ci_real_data = [
    "inference_data_constructor_seen_citation_model_data"
]
citation_model_data_fixtures_seen_slow_skip_ci = (
    citation_model_data_fixtures_seen_slow_skip_ci_real_data
    + citation_model_data_fixtures_seen_slow_skip_ci_test_data
)

citation_model_data_fixtures_unseen_skip_ci_test_data = [
    "unseen_paper_attribute_getter_citation_model_data"
]
citation_model_data_fixtures_unseen_skip_ci_real_data = [
    "inference_data_constructor_unseen_citation_model_data"
]
citation_model_data_fixtures_unseen_skip_ci = (
    citation_model_data_fixtures_unseen_skip_ci_real_data
    + citation_model_data_fixtures_unseen_skip_ci_test_data
)


@pytest.mark.parametrize(
    "model_data",
    [
        *[lazy_fixture(fixture) for fixture in citation_model_data_fixtures_seen],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in citation_model_data_fixtures_seen_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in citation_model_data_fixtures_unseen_skip_ci
        ],
    ],
)
def test_citation_model_data_getitem(model_data: CitationModelData) -> None:
    index_info_matrix = model_data.info_frame.index
    index_feature_matrix = model_data.features_frame.index
    shared_indices = index_info_matrix.intersection(index_feature_matrix)

    # check that slicing works for info matrix, feature matrix and integer labels
    sliced_model_data = model_data[shared_indices]
    assert isinstance(sliced_model_data, CitationModelData)

    assert len(sliced_model_data.info_frame) == len(shared_indices)
    assert len(sliced_model_data.integer_labels_frame) == len(shared_indices)
    assert len(sliced_model_data.features_frame) == len(shared_indices)


@pytest.mark.parametrize(
    "model_data",
    [
        *[lazy_fixture(fixture) for fixture in citation_model_data_fixtures_seen],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in citation_model_data_fixtures_seen_slow_skip_ci_test_data
        ],
    ],
)
def test_seen_citation_model_data_getitem(
    model_data: CitationModelData, test_data_size: int
) -> None:
    index_info_matrix = model_data.info_frame.index
    # -1 since query document is excluded from candidates
    assert len(index_info_matrix) == test_data_size - 1

    index_feature_matrix = model_data.features_frame.index
    assert len(index_feature_matrix) == test_data_size - 1

    # index of info matrix and feature matrix is identical
    shared_indices = index_info_matrix.intersection(index_feature_matrix)
    assert len(shared_indices) == test_data_size - 1


@pytest.mark.parametrize(
    "model_data",
    [
        pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
        for fixture in citation_model_data_fixtures_unseen_skip_ci_test_data
    ],
)
def test_unseen_citation_model_data_getitem(
    model_data: CitationModelData, test_data_size: int
) -> None:
    index_info_matrix = model_data.info_frame.index
    assert len(index_info_matrix) == test_data_size

    index_feature_matrix = model_data.features_frame.index
    assert len(index_feature_matrix) == test_data_size

    shared_indices = index_info_matrix.intersection(index_feature_matrix)
    assert len(shared_indices) == test_data_size


# SUBSECTION: Test Info Matrix
citation_model_data_info_matrix_fixtures_seen = [
    "citation_model_data_info_matrix",
    "citation_model_data_constructor_info_matrix",
]

citation_model_data_info_matrix_fixtures_seen_slow_skip_ci = [
    "seen_paper_attribute_getter_citation_model_data_info_matrix",
    "inference_data_constructor_seen_citation_model_data_info_matrix",
]

citation_model_data_info_matrix_fixtures_unseen_skip_ci = [
    "unseen_paper_attribute_getter_citation_model_data_info_matrix",
    "inference_data_constructor_unseen_citation_model_data_info_matrix",
]


@pytest.mark.parametrize(
    "info_matrix",
    [
        *[lazy_fixture(fixture) for fixture in citation_model_data_info_matrix_fixtures_seen],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in citation_model_data_info_matrix_fixtures_seen_slow_skip_ci
        ],
        *[
            pytest.param(
                lazy_fixture(fixture),
                marks=(pytest.mark.skip_ci),
            )
            for fixture in citation_model_data_info_matrix_fixtures_unseen_skip_ci
        ],
    ],
)
def test_citation_model_data_info_matrix(info_matrix: pl.DataFrame) -> None:
    assert isinstance(info_matrix, pl.DataFrame)

    assert info_matrix.index.name == "document_id"
    assert info_matrix.index.dtype == pl.Int64Dtype()

    assert info_matrix.shape[1] == 8
    assert info_matrix.columns.to_list() == [
        "title",
        "author",
        "arxiv_labels",
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis",
        "bibliographic_coupling",
    ]
    assert info_matrix.dtypes.to_list() == [
        pl.StringDtype(),
        pl.StringDtype(),
        np.dtype("O"),
        pl.StringDtype(),
        pl.Int64Dtype(),
        pl.Int64Dtype(),
        np.dtype("int64"),
        np.dtype("int64"),
    ]


# SUBSECTION: Test Feature Matrix
citation_model_data_feature_matrix_fixtures_seen = [
    "citation_model_data_feature_matrix",
    "citation_model_data_constructor_feature_matrix",
]

citation_model_data_feature_matrix_fixtures_seen_slow_skip_ci = [
    "seen_paper_attribute_getter_citation_model_data_feature_matrix",
    "inference_data_constructor_seen_citation_model_data_feature_matrix",
]

citation_model_data_feature_matrix_fixtures_unseen_skip_ci = [
    "unseen_paper_attribute_getter_citation_model_data_feature_matrix",
    "inference_data_constructor_unseen_citation_model_data_feature_matrix",
]


@pytest.mark.parametrize(
    "feature_matrix",
    [
        *[lazy_fixture(fixture) for fixture in citation_model_data_feature_matrix_fixtures_seen],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in citation_model_data_feature_matrix_fixtures_seen_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in citation_model_data_feature_matrix_fixtures_unseen_skip_ci
        ],
    ],
)
def test_citation_model_data_feature_matrix(feature_matrix: pl.DataFrame) -> None:
    assert isinstance(feature_matrix, pl.DataFrame)

    assert feature_matrix.index.name == "document_id"
    assert feature_matrix.index.dtype == pl.Int64Dtype()

    assert feature_matrix.shape[1] == 5
    assert feature_matrix.columns.to_list() == [
        "publication_date_rank",
        "citationcount_document_rank",
        "citationcount_author_rank",
        "co_citation_analysis_rank",
        "bibliographic_coupling_rank",
    ]
    assert all(
        feature_matrix.dtypes == [np.float64, np.float64, np.float64, np.float64, np.float64]
    )
