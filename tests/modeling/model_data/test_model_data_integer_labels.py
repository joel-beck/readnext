import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils.aliases import IntegerLabelsFrame

integer_labels_fixtures_seen = [
    lazy_fixture("model_data_constructor_seen_integer_labels_frame"),
    lazy_fixture("model_data_seen_integer_labels_frame"),
]

integer_labels_fixtures_unseen = [
    lazy_fixture("model_data_constructor_unseen_integer_labels_frame"),
    lazy_fixture("model_data_unseen_integer_labels_frame"),
]

integer_labels_fixtures_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_seen_model_data_integer_labels_frame"),
    lazy_fixture("inference_data_constructor_seen_model_data_integer_labels_frame"),
]

integer_labels_fixtures_slow_skip_ci = [
    lazy_fixture("inference_data_constructor_plugin_unseen_model_data_integer_labels_frame"),
    lazy_fixture("inference_data_constructor_unseen_model_data_integer_labels_frame"),
]


@pytest.mark.parametrize(
    "integer_labels_frame",
    [
        *[
            pytest.param(fixture)
            for fixture in integer_labels_fixtures_seen + integer_labels_fixtures_unseen
        ],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in integer_labels_fixtures_skip_ci
        ],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in integer_labels_fixtures_slow_skip_ci
        ],
    ],
)
def test_integer_labels_frame(
    integer_labels_frame: IntegerLabelsFrame,
) -> None:
    assert isinstance(integer_labels_frame, pl.DataFrame)

    assert integer_labels_frame.shape[1] == 2
    assert integer_labels_frame.columns == [
        "candidate_d3_document_id",
        "integer_label",
    ]

    assert integer_labels_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert integer_labels_frame["integer_label"].dtype == pl.Int64


@pytest.mark.parametrize(
    "integer_labels_frame",
    [
        *[pytest.param(fixture) for fixture in integer_labels_fixtures_seen],
        *[
            pytest.param(fixture, marks=pytest.mark.skip_ci)
            for fixture in integer_labels_fixtures_skip_ci
        ],
    ],
)
def test_integer_labels_frame_seen(
    integer_labels_frame: IntegerLabelsFrame,
) -> None:
    # check that all integer labels are either 0 or 1
    assert integer_labels_frame["integer_label"].is_in([0, 1]).all()


@pytest.mark.parametrize(
    "integer_labels_frame",
    [
        *[pytest.param(fixture) for fixture in integer_labels_fixtures_unseen],
        *[
            pytest.param(fixture, marks=[pytest.mark.slow, pytest.mark.skip_ci])
            for fixture in integer_labels_fixtures_slow_skip_ci
        ],
    ],
)
def test_integer_labels_frame_unseen(
    integer_labels_frame: IntegerLabelsFrame,
) -> None:
    # check that all integer labels are 0 for unseen documents
    assert (integer_labels_frame["integer_label"] == 0).all()
