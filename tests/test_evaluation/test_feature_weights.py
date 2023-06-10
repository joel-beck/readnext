import pytest
from pydantic import ValidationError

from readnext.evaluation.scoring import FeatureWeights


def test_pydantic_validation() -> None:
    # Correct data
    fw = FeatureWeights(1.0, 1.0, 1.0, 1.0, 1.0)
    assert fw.publication_date == 1.0
    assert fw.citationcount_document == 1.0
    assert fw.citationcount_author == 1.0
    assert fw.co_citation_analysis == 1.0
    assert fw.bibliographic_coupling == 1.0

    with pytest.raises(ValidationError):
        fw = FeatureWeights(-1.0, 1.0, 1.0, 1.0, 1.0)

    with pytest.raises(ValidationError):
        fw = FeatureWeights("incorrect_type", 1.0, 1.0, 1.0, 1.0)  # type: ignore


def test_feature_weights_normalization() -> None:
    fw = FeatureWeights(1.0, 1.0, 1.0, 1.0, 1.0)
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

    fw = FeatureWeights(1.0, 2.0, 3.0, 4.0, 5.0)
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
