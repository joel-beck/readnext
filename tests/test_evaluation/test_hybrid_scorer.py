import pytest
import pandas as pd

from readnext.evaluation.scoring import HybridScorer
from readnext.modeling.document_info import DocumentInfo
from readnext.modeling.model_data import LanguageModelData, CitationModelData


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
                info_matrix=pd.DataFrame(),
                integer_labels=pd.Series(),
                cosine_similarity_ranks=pd.DataFrame(),
            ),
            CitationModelData(
                query_document=DocumentInfo(
                    d3_document_id=-1,
                    title="Title",
                    author="Author",
                    abstract="Abstract",
                    arxiv_labels=[],
                ),
                info_matrix=pd.DataFrame(),
                feature_matrix=pd.DataFrame(),
                integer_labels=pd.Series(),
            ),
        )
