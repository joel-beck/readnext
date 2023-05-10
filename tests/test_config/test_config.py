from pathlib import Path

from readnext.config import (
    D3,
    ArxivDataPaths,
    CitationModelsResultsPaths,
    D3AuthorsDataPaths,
    D3DocumentsDataPaths,
    DataPaths,
    LanguageModelsResultsPaths,
    MergedDataPaths,
    ModelPaths,
    ModelVersions,
    ResultsPaths,
)


def test_d3_documents_data_paths() -> None:
    d3_documents_data_paths = D3DocumentsDataPaths()
    assert isinstance(d3_documents_data_paths.raw_json, Path)
    assert isinstance(d3_documents_data_paths.chunks_stem, Path)
    assert isinstance(d3_documents_data_paths.full_pkl, Path)
    assert isinstance(d3_documents_data_paths.preprocessed_chunks_stem, Path)


def test_d3_authors_data_paths() -> None:
    d3_authors_data_paths = D3AuthorsDataPaths()
    assert isinstance(d3_authors_data_paths.raw_json, Path)
    assert isinstance(d3_authors_data_paths.most_cited_pkl, Path)
    assert isinstance(d3_authors_data_paths.full_pkl, Path)


def test_d3() -> None:
    d3 = D3()
    assert isinstance(d3.documents, D3DocumentsDataPaths)
    assert isinstance(d3.authors, D3AuthorsDataPaths)


def test_arxiv_data_paths() -> None:
    arxiv_data_paths = ArxivDataPaths()
    assert isinstance(arxiv_data_paths.raw_json, Path)
    assert isinstance(arxiv_data_paths.id_labels_pkl, Path)


def test_merged_data_paths() -> None:
    merged_data_paths = MergedDataPaths()
    assert isinstance(merged_data_paths.documents_labels_chunk_stem, Path)
    assert isinstance(merged_data_paths.documents_labels_pkl, Path)
    assert isinstance(merged_data_paths.documents_authors_labels_pkl, Path)
    assert isinstance(merged_data_paths.documents_authors_labels_citations_chunks_stem, Path)
    assert isinstance(merged_data_paths.documents_authors_labels_citations_pkl, Path)
    assert isinstance(merged_data_paths.documents_authors_labels_citations_most_cited_pkl, Path)
    assert isinstance(merged_data_paths.most_cited_subset_size, int)


def test_data_paths() -> None:
    data_paths = DataPaths()
    assert isinstance(data_paths.d3, D3)
    assert isinstance(data_paths.arxiv, ArxivDataPaths)
    assert isinstance(data_paths.merged, MergedDataPaths)


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
    assert isinstance(
        citation_models_results_paths.bibliographic_coupling_scores_most_cited_pkl, Path
    )
    assert isinstance(
        citation_models_results_paths.co_citation_analysis_scores_most_cited_pkl, Path
    )


def test_language_models_results_paths() -> None:
    language_models_results_paths = LanguageModelsResultsPaths()
    assert isinstance(
        language_models_results_paths.spacy_tokenized_abstracts_mapping_most_cited_pkl, Path
    )
    assert isinstance(language_models_results_paths.tfidf_embeddings_mapping_most_cited_pkl, Path)
    assert isinstance(language_models_results_paths.tfidf_cosine_similarities_most_cited_pkl, Path)
    assert isinstance(
        language_models_results_paths.word2vec_embeddings_mapping_most_cited_pkl, Path
    )
    assert isinstance(
        language_models_results_paths.word2vec_cosine_similarities_most_cited_pkl, Path
    )
    assert isinstance(language_models_results_paths.glove_embeddings_mapping_most_cited_pkl, Path)
    assert isinstance(language_models_results_paths.glove_cosine_similarities_most_cited_pkl, Path)
    assert isinstance(
        language_models_results_paths.fasttext_embeddings_mapping_most_cited_pkl, Path
    )
    assert isinstance(
        language_models_results_paths.fasttext_cosine_similarities_most_cited_pkl, Path
    )
    assert isinstance(
        language_models_results_paths.bert_tokenized_abstracts_mapping_most_cited_pkl, Path
    )
    assert isinstance(language_models_results_paths.bert_embeddings_mapping_most_cited_pkl, Path)
    assert isinstance(language_models_results_paths.bert_cosine_similarities_most_cited_pkl, Path)
    assert isinstance(
        language_models_results_paths.scibert_tokenized_abstracts_mapping_most_cited_pkl, Path
    )
    assert isinstance(language_models_results_paths.scibert_embeddings_mapping_most_cited_pkl, Path)
    assert isinstance(
        language_models_results_paths.scibert_cosine_similarities_most_cited_pkl, Path
    )


def test_results_paths() -> None:
    results_paths = ResultsPaths()
    assert isinstance(results_paths.citation_models, CitationModelsResultsPaths)
    assert isinstance(results_paths.language_models, LanguageModelsResultsPaths)
