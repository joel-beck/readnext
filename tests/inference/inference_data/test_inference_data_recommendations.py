import polars as pl
import pytest

from readnext.inference.features import Recommendations


def test_kw_only_initialization_recommendations() -> None:
    with pytest.raises(TypeError):
        Recommendations(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
