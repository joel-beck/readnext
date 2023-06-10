"""
Specifies placeholder default values for quick object initialization.
"""

import polars as pl

from readnext.config import DataPaths
from readnext.evaluation.scoring import FeatureWeights
from readnext.inference.constructor_plugin_seen import (
    SeenInferenceDataConstructorPlugin,
)
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice

documents_frame_default = pl.scan_parquet(DataPaths.merged.documents_frame).head(10).collect()

document_info_default = DocumentInfo(
    d3_document_id=-1,
    title="Title",
    author="Author",
    arxiv_labels=["cs.CL"],
    semanticscholar_url="https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
    arxiv_url="https://arxiv.org/abs/2106.01572",
    abstract="Abstract",
)

seen_inference_data_constructor_plugin_default = SeenInferenceDataConstructorPlugin(
    semanticscholar_url="https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
    language_model_choice=LanguageModelChoice.TFIDF,
    feature_weights=FeatureWeights(),
    documents_frame=documents_frame_default,
)

citation_model_data_default = CitationModelData(
    query_document=document_info_default,
    info_frame=pl.DataFrame(),
    features_frame=pl.DataFrame(),
    integer_labels_frame=pl.DataFrame(),
    ranks_frame=pl.DataFrame(),
    points_frame=pl.DataFrame(),
)

language_model_data_default = LanguageModelData(
    query_document=document_info_default,
    info_frame=pl.DataFrame(),
    features_frame=pl.DataFrame(),
    integer_labels_frame=pl.DataFrame(),
)
