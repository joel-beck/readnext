import numpy as np
import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling import LanguageModelData

# SUBSECTION: Test Getitem
language_model_data_fixtures_seen = ["language_model_data"]

language_model_data_fixtures_seen_slow_skip_ci_test_data = [
    "seen_paper_attribute_getter_language_model_data"
]
language_model_data_fixtures_seen_slow_skip_ci_real_data = [
    "inference_data_constructor_seen_language_model_data"
]
language_model_data_fixtures_seen_slow_skip_ci = (
    language_model_data_fixtures_seen_slow_skip_ci_real_data
    + language_model_data_fixtures_seen_slow_skip_ci_test_data
)

language_model_data_fixtures_unseen_skip_ci_test_data = [
    "unseen_paper_attribute_getter_language_model_data"
]
language_model_data_fixtures_unseen_skip_ci_real_data = [
    "inference_data_constructor_unseen_language_model_data"
]
language_model_data_fixtures_unseen_skip_ci = (
    language_model_data_fixtures_unseen_skip_ci_real_data
    + language_model_data_fixtures_unseen_skip_ci_test_data
)


@pytest.mark.parametrize(
    "model_data",
    [
        *[lazy_fixture(fixture) for fixture in language_model_data_fixtures_seen],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in language_model_data_fixtures_seen_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in language_model_data_fixtures_unseen_skip_ci
        ],
    ],
)
def test_language_model_data_getitem(model_data: LanguageModelData) -> None:
    index_info_matrix = model_data.info_matrix.index
    index_cosine_similarity_ranks = model_data.cosine_similarity_ranks.index
    shared_indices = index_info_matrix.intersection(index_cosine_similarity_ranks)

    # check that slicing works for info matrix, cosine similarity ranks matrix and integer labels
    sliced_model_data = model_data[shared_indices]
    assert isinstance(sliced_model_data, LanguageModelData)

    assert len(sliced_model_data.info_matrix) == len(shared_indices)
    assert len(sliced_model_data.integer_labels) == len(shared_indices)
    assert len(sliced_model_data.cosine_similarity_ranks) == len(shared_indices)


@pytest.mark.parametrize(
    "model_data",
    [
        *[lazy_fixture(fixture) for fixture in language_model_data_fixtures_seen],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in language_model_data_fixtures_seen_slow_skip_ci_test_data
        ],
    ],
)
def test_seen_language_model_data_getitem(
    model_data: LanguageModelData, test_data_size: int
) -> None:
    index_info_matrix = model_data.info_matrix.index
    assert len(index_info_matrix) == test_data_size - 1

    index_cosine_similarity_ranks = model_data.cosine_similarity_ranks.index
    # the cosine similarities test data frame has 100 rows, but each row still contains
    # a list of the original 1000 `DocumentScore` objects
    assert len(index_cosine_similarity_ranks) == 999

    # index of info matrix is a strict subset of cosine similarity ranks index
    shared_indices = index_info_matrix.intersection(index_cosine_similarity_ranks)
    assert len(shared_indices) == test_data_size - 1


@pytest.mark.parametrize(
    "model_data",
    [
        pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
        for fixture in language_model_data_fixtures_unseen_skip_ci_test_data
    ],
)
def test_unseen_language_model_data_getitem(
    model_data: LanguageModelData, test_data_size: int
) -> None:
    index_info_matrix = model_data.info_matrix.index
    assert len(index_info_matrix) == test_data_size

    index_cosine_similarity_ranks = model_data.cosine_similarity_ranks.index
    assert len(index_cosine_similarity_ranks) == 1000

    shared_indices = index_info_matrix.intersection(index_cosine_similarity_ranks)
    assert len(shared_indices) == test_data_size


# SUBSECTION: Test Info Matrix
language_model_data_info_matrix_fixtures_seen = [
    "language_model_data_info_matrix",
    "language_model_data_constructor_info_matrix",
]

language_model_data_info_matrix_fixtures_seen_slow_skip_ci = [
    "seen_paper_attribute_getter_language_model_data_info_matrix",
    "inference_data_constructor_seen_language_model_data_info_matrix",
]

language_model_data_info_matrix_fixtures_unseen_skip_ci = [
    "unseen_paper_attribute_getter_language_model_data_info_matrix",
    "inference_data_constructor_unseen_language_model_data_info_matrix",
]


@pytest.mark.parametrize(
    "info_matrix",
    [
        *[lazy_fixture(fixture) for fixture in language_model_data_info_matrix_fixtures_seen],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in language_model_data_info_matrix_fixtures_seen_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in language_model_data_info_matrix_fixtures_unseen_skip_ci
        ],
    ],
)
def test_language_model_data_info_matrix(info_matrix: pl.DataFrame) -> None:
    assert isinstance(info_matrix, pl.DataFrame)

    assert info_matrix.index.name == "document_id"
    assert info_matrix.index.dtype == pl.Int64Dtype()

    assert info_matrix.shape[1] == 4
    assert info_matrix.columns.to_list() == [
        "title",
        "author",
        "arxiv_labels",
        "cosine_similarity",
    ]
    assert info_matrix.dtypes.to_list() == [
        pl.StringDtype(),
        pl.StringDtype(),
        np.dtype("O"),
        np.dtype("float64"),
    ]


# SUBSECTION: Test Cosine Similarity Ranks
language_model_data_cosine_similarity_ranks_fixtures_seen = [
    "language_model_data_cosine_similarity_ranks",
    "language_model_data_constructor_cosine_similarity_ranks",
]

language_model_data_cosine_similarity_ranks_fixtures_seen_slow_skip_ci = [
    "seen_paper_attribute_getter_language_model_data_cosine_similarity_ranks",
    "inference_data_constructor_seen_language_model_data_cosine_similarity_ranks",
]

language_model_data_cosine_similarity_ranks_fixtures_unseen_skip_ci = [
    "unseen_paper_attribute_getter_language_model_data_cosine_similarity_ranks",
    "inference_data_constructor_unseen_language_model_data_cosine_similarity_ranks",
]


@pytest.mark.parametrize(
    "cosine_similarity_ranks",
    [
        *[
            lazy_fixture(fixture)
            for fixture in language_model_data_cosine_similarity_ranks_fixtures_seen
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in language_model_data_cosine_similarity_ranks_fixtures_seen_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in language_model_data_cosine_similarity_ranks_fixtures_unseen_skip_ci
        ],
    ],
)
def test_language_model_data_cosine_similarity_ranks(
    cosine_similarity_ranks: pl.DataFrame,
) -> None:
    assert isinstance(cosine_similarity_ranks, pl.DataFrame)

    assert cosine_similarity_ranks.index.name == "document_id"
    assert cosine_similarity_ranks.index.dtype == np.int64

    assert cosine_similarity_ranks.shape[1] == 1
    assert cosine_similarity_ranks.columns.to_list() == ["cosine_similarity_rank"]
    assert all(cosine_similarity_ranks.dtypes == [np.float64])

    # check that lowest / best rank is 1.0
    assert min(cosine_similarity_ranks["cosine_similarity_rank"]) == 1.0

    # check that no rank is higher than the number of documents
    assert max(cosine_similarity_ranks["cosine_similarity_rank"]) <= len(cosine_similarity_ranks)
