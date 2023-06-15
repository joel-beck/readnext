import polars as pl
import pytest

from readnext.inference.features import Labels


def test_kw_only_initialization_labels() -> None:
    with pytest.raises(TypeError):
        Labels(pl.DataFrame(), pl.DataFrame())  # type: ignore
