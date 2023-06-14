import re

import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier
from readnext.inference.features import Features, Labels, Points, Ranks, Recommendations


def test_kw_only_initialization_points() -> None:
    with pytest.raises(TypeError):
        Points(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
