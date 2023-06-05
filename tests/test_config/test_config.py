from pathlib import Path

from readnext.config import (
    CitationModelsResultsPaths,
    DataPaths,
    LanguageModelsResultsPaths,
    MagicNumbers,
    MergedDataPaths,
    ModelPaths,
    ModelVersions,
    RawDataPaths,
    ResultsPaths,
)


def test_raw_data_paths() -> None:
    raw_data_paths = RawDataPaths()
    assert isinstance(raw_data_paths.documents_json, Path)
    assert isinstance(raw_data_paths.documents_parquet, Path)
    assert isinstance(raw_data_paths.authors_json, Path)
    assert isinstance(raw_data_paths.authors_parquet, Path)
    assert isinstance(raw_data_paths.arxiv_labels_json, Path)
    assert isinstance(raw_data_paths.arxiv_labels_parquet, Path)


def test_merged_data_paths() -> None:
    merged_data_paths = MergedDataPaths()
    assert isinstance(merged_data_paths.documents_labels, Path)
    assert isinstance(merged_data_paths.documents_authors_labels, Path)
    assert isinstance(merged_data_paths.documents_authors_labels_citations, Path)
    assert isinstance(merged_data_paths.documents_data, Path)


def test_data_paths() -> None:
    data_paths = DataPaths()
    assert isinstance(data_paths.raw, RawDataPaths)
    assert isinstance(data_paths.merged, MergedDataPaths)


def test_magic_numbers() -> None:
    magic_numbers = MagicNumbers()
    assert isinstance(magic_numbers.documents_data_intermediate_cutoff, int)
    assert isinstance(magic_numbers.documents_data_final_size, int)
    assert isinstance(magic_numbers.scoring_limit, int)
    assert isinstance(magic_numbers.n_candidates, int)
    assert isinstance(magic_numbers.n_recommendations, int)


def test_model_versions() -> None:
    model_versions = ModelVersions()
    assert isinstance(model_versions.spacy, str)
    assert isinstance(model_versions.word2vec, str)
    assert isinstance(model_versions.glove, str)
    assert isinstance(model_versions.fasttext, str)
    assert isinstance(model_versions.bert, str)
    assert isinstance(model_versions.scibert, str)


def test_model_paths() -> None:
    model_paths = ModelPaths()
    assert isinstance(model_paths.word2vec, Path)
    assert isinstance(model_paths.glove, Path)
    assert isinstance(model_paths.fasttext, Path)


def test_citation_models_results_paths() -> None:
    citation_models_results_paths = CitationModelsResultsPaths()
    assert isinstance(citation_models_results_paths.bibliographic_coupling_scores_parquet, Path)
    assert isinstance(citation_models_results_paths.co_citation_analysis_scores_parquet, Path)


def test_language_models_results_paths() -> None:
    language_models_results_paths = LanguageModelsResultsPaths()
    assert isinstance(language_models_results_paths.spacy_tokenized_abstracts_parquet, Path)
    assert isinstance(language_models_results_paths.tfidf_embeddings_parquet, Path)
    assert isinstance(language_models_results_paths.tfidf_cosine_similarities_parquet, Path)
    assert isinstance(language_models_results_paths.word2vec_embeddings_parquet, Path)
    assert isinstance(language_models_results_paths.word2vec_cosine_similarities_parquet, Path)
    assert isinstance(language_models_results_paths.glove_embeddings_parquet, Path)
    assert isinstance(language_models_results_paths.glove_cosine_similarities_parquet, Path)
    assert isinstance(language_models_results_paths.fasttext_embeddings_parquet, Path)
    assert isinstance(language_models_results_paths.fasttext_cosine_similarities_parquet, Path)
    assert isinstance(language_models_results_paths.bert_tokenized_abstracts_parquet, Path)
    assert isinstance(language_models_results_paths.bert_embeddings_parquet, Path)
    assert isinstance(language_models_results_paths.bert_cosine_similarities_parquet, Path)
    assert isinstance(language_models_results_paths.scibert_tokenized_abstracts_parquet, Path)
    assert isinstance(language_models_results_paths.scibert_embeddings_parquet, Path)
    assert isinstance(language_models_results_paths.scibert_cosine_similarities_parquet, Path)


def test_results_paths() -> None:
    results_paths = ResultsPaths()
    assert isinstance(results_paths.citation_models, CitationModelsResultsPaths)
    assert isinstance(results_paths.language_models, LanguageModelsResultsPaths)
