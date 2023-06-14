import dataclasses

import pytest
from pydantic import ValidationError

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference.features import Features


@pytest.mark.updated
def test_from_inference_data(inference_data_features: Features) -> None:
    assert isinstance(inference_data_features.feature_weights, FeatureWeights)
    assert list(dataclasses.asdict(inference_data_features.feature_weights)) == [
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis",
        "bibliographic_coupling",
    ]

    assert isinstance(inference_data_features.feature_weights.publication_date, float)
    assert isinstance(inference_data_features.feature_weights.citationcount_document, float)
    assert isinstance(inference_data_features.feature_weights.citationcount_author, float)
    assert isinstance(inference_data_features.feature_weights.co_citation_analysis, float)
    assert isinstance(inference_data_features.feature_weights.bibliographic_coupling, float)


@pytest.mark.updated
def test_pydantic_validation() -> None:
    # Correct data
    fw = FeatureWeights(
        publication_date=1.0,
        citationcount_document=1.0,
        citationcount_author=1.0,
        co_citation_analysis=1.0,
        bibliographic_coupling=1.0,
    )
    assert fw.publication_date == 1.0
    assert fw.citationcount_document == 1.0
    assert fw.citationcount_author == 1.0
    assert fw.co_citation_analysis == 1.0
    assert fw.bibliographic_coupling == 1.0

    with pytest.raises(ValidationError):
        fw = FeatureWeights(
            publication_date=-1.0,
            citationcount_document=1.0,
            citationcount_author=1.0,
            co_citation_analysis=1.0,
            bibliographic_coupling=1.0,
        )

    with pytest.raises(ValidationError):
        fw = FeatureWeights(
            publication_date="incorrect_type",  # type: ignore
            citationcount_document=1.0,
            citationcount_author=1.0,
            co_citation_analysis=1.0,
            bibliographic_coupling=1.0,
        )


@pytest.mark.updated
def test_feature_weights_normalization() -> None:
    fw = FeatureWeights(
        publication_date=1.0,
        citationcount_document=1.0,
        citationcount_author=1.0,
        co_citation_analysis=1.0,
        bibliographic_coupling=1.0,
    )
    normalized_fw = fw.normalize()

    assert (
        sum(
            [
                normalized_fw.publication_date,
                normalized_fw.citationcount_document,
                normalized_fw.citationcount_author,
                normalized_fw.co_citation_analysis,
                normalized_fw.bibliographic_coupling,
            ]
        )
        == 1.0
    )

    fw = FeatureWeights(
        publication_date=-1.0,
        citationcount_document=2.0,
        citationcount_author=3.0,
        co_citation_analysis=4.0,
        bibliographic_coupling=5.0,
    )

    normalized_fw = fw.normalize()
    assert (
        sum(
            [
                normalized_fw.publication_date,
                normalized_fw.citationcount_document,
                normalized_fw.citationcount_author,
                normalized_fw.co_citation_analysis,
                normalized_fw.bibliographic_coupling,
            ]
        )
        == 1.0
    )


@pytest.mark.updated
def test_kw_only_initialization_feature_weights() -> None:
    with pytest.raises(TypeError):
        FeatureWeights(1.0, 1.0, 1.0, 1.0, 1.0)  # type: ignore
