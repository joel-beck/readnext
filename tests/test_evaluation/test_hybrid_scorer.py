import polars as pl
import pytest

from readnext.evaluation.scoring import HybridScorer
from readnext.modeling.document_info import DocumentInfo
from readnext.modeling.model_data import CitationModelData, LanguageModelData


def test_kw_only_initialization_hybrid_scorer() -> None:
    with pytest.raises(TypeError):
        HybridScorer(
            "TF-IDF",  # type: ignore
            LanguageModelData(
                query_document=DocumentInfo(
                    d3_document_id=-1,
                    title="Title",
                    author="Author",
                    abstract="Abstract",
                    arxiv_labels=[],
                ),
                info_frame=pl.DataFrame(),
                integer_labels_frame=pl.DataFrame(),
                cosine_similarity_ranks=pl.DataFrame(),
            ),
            CitationModelData(
                query_document=DocumentInfo(
                    d3_document_id=-1,
                    title="Title",
                    author="Author",
                    abstract="Abstract",
                    arxiv_labels=[],
                ),
                info_frame=pl.DataFrame(),
                features_frame=pl.DataFrame(),
                integer_labels_frame=pl.DataFrame(),
            ),
        )
