import polars as pl
import pytest

from readnext.evaluation.scoring import HybridScorer
from readnext.modeling.document_info import DocumentInfo
from readnext.modeling.model_data import CitationModelData, LanguageModelData


@pytest.mark.updated
def test_kw_only_initialization_hybrid_scorer() -> None:
    with pytest.raises(TypeError):
        HybridScorer(
            "TF-IDF",  # type: ignore
            LanguageModelData(
                query_document=DocumentInfo(
                    d3_document_id=-1,
                    title="Title",
                    author="Author",
                    publication_date="2000-01-01",
                    semanticscholar_url="",
                    arxiv_url="",
                    abstract="Abstract",
                    arxiv_labels=[],
                ),
                info_frame=pl.DataFrame(),
                integer_labels_frame=pl.DataFrame(),
                features_frame=pl.DataFrame(),
            ),
            CitationModelData(
                query_document=DocumentInfo(
                    d3_document_id=-1,
                    title="Title",
                    author="Author",
                    publication_date="2000-01-01",
                    abstract="Abstract",
                    arxiv_labels=[],
                ),
                info_frame=pl.DataFrame(),
                features_frame=pl.DataFrame(),
                integer_labels_frame=pl.DataFrame(),
                ranks_frame=pl.DataFrame(),
                points_frame=pl.DataFrame(),
            ),
        )
