import dataclasses

import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.inference import DocumentIdentifier
from readnext.inference.features import Features, Labels, Points, Ranks, Recommendations
from readnext.inference.inference_data import InferenceData
from readnext.modeling import DocumentInfo

inference_data_fixtures_skip_ci = [lazy_fixture("inference_data_seen")]

inference_data_fixtures_slow_skip_ci = [lazy_fixture("inference_data_unseen")]


@pytest.mark.updated
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "inference_data",
    [
        *[pytest.param(fixture) for fixture in inference_data_fixtures_skip_ci],
        *[
            pytest.param(fixture, marks=pytest.mark.slow)
            for fixture in inference_data_fixtures_slow_skip_ci
        ],
    ],
)
def test_inference_data_attributes(inference_data: InferenceData) -> None:
    assert isinstance(inference_data, InferenceData)

    assert list(dataclasses.asdict(inference_data)) == [
        "document_identifier",
        "document_info",
        "features",
        "ranks",
        "points",
        "labels",
        "recommendations",
    ]

    assert isinstance(inference_data.document_info, DocumentInfo)
    assert isinstance(inference_data.document_identifier, DocumentIdentifier)
    assert isinstance(inference_data.features, Features)
    assert isinstance(inference_data.ranks, Ranks)
    assert isinstance(inference_data.points, Points)
    assert isinstance(inference_data.labels, Labels)
    assert isinstance(inference_data.recommendations, Recommendations)


@pytest.mark.updated
def test_kw_only_initialization_inference_data(
    dummy_document_identifier: DocumentIdentifier,
    dummy_document_info: DocumentInfo,
    dummy_features: Features,
    dummy_ranks: Ranks,
    dummy_points: Points,
    dummy_labels: Labels,
    dummy_recommendations: Recommendations,
) -> None:
    with pytest.raises(TypeError):
        InferenceData(
            dummy_document_identifier,  # type: ignore
            dummy_document_info,
            dummy_features,
            dummy_ranks,
            dummy_points,
            dummy_labels,
            dummy_recommendations,
        )
