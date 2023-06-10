import polars as pl
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier
from readnext.inference.constructor import Features, Labels, Ranks, Recommendations


def test_kw_only_initialization_document_identifier() -> None:
    with pytest.raises(TypeError):
        DocumentIdentifier(
            -1,  # type: ignore
            "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
            "2303.08774",
            "https://arxiv.org/abs/2303.08774",
        )


def test_kw_only_initialization_features() -> None:
    with pytest.raises(TypeError):
        Features(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            FeatureWeights(),
        )


def test_kw_only_initialization_ranks() -> None:
    with pytest.raises(TypeError):
        Ranks(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )


def test_kw_only_initialization_labels() -> None:
    with pytest.raises(TypeError):
        Labels(pl.DataFrame(), pl.DataFrame())  # type: ignore


def test_kw_only_initialization_recommendations() -> None:
    with pytest.raises(TypeError):
        Recommendations(
            pl.DataFrame(),  # type: ignore
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
        )
