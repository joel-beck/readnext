"""
- Set file paths for reading and writing data
- Set file paths for reading pretrained models
- Set file paths for reading and writing model results
- Specify Model Versions

Requires a .env file in the root directory with the following variables:
- DATA_DIRPATH (default: `data`): path to the data directory where all data files are
stored
- MODELS_DIRPATH (default: `models`): path to the models directory where all pretrained
models are stored
- RESULTS_DIRPATH (default: `results`): path to the results directory where all model
results are stored
- DOCUMENTS_METADATA_FILENAME (default: `2022-11-30-papers.jsonl`): name of the file
containing the downloaded D3 documents in jsonl format
- AUTHORS_METADATA_FILENAME (default: `2022-11-30-authors.jsonl`): name of the file
containing the downloaded D3 authors in jsonl format
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

data_dirpath = Path(os.getenv("DATA_DIRPATH", "data"))
models_dirpath = Path(os.getenv("MODELS_DIRPATH", "models"))
results_dirpath = Path(os.getenv("RESULTS_DIRPATH", "results"))

documents_metadata_json_filename = os.getenv(
    "DOCUMENTS_METADATA_FILENAME", "2022-11-30-papers.jsonl"
)
authors_metadata_json_filename = os.getenv("AUTHORS_METADATA_FILENAME", "2022-11-30-authors.jsonl")


@dataclass(frozen=True)
class D3DocumentsDataPaths:
    raw_json: Path = data_dirpath / documents_metadata_json_filename
    chunks_stem: Path = data_dirpath / "2022-11-30_documents_chunks"
    full_pkl: Path = data_dirpath / "2022-11-30_documents_full.pkl"
    preprocessed_chunks_stem: Path = data_dirpath / "documents_preprocessed_chunks"


@dataclass(frozen=True)
class D3AuthorsDataPaths:
    raw_json: Path = data_dirpath / authors_metadata_json_filename
    most_cited_pkl: Path = data_dirpath / "2022-11-30_authors_most_cited.pkl"
    full_pkl: Path = data_dirpath / "2022-11-30_authors_full.pkl"


@dataclass(frozen=True)
class D3:
    documents: D3DocumentsDataPaths = D3DocumentsDataPaths()
    authors: D3AuthorsDataPaths = D3AuthorsDataPaths()


@dataclass(frozen=True)
class ArxivDataPaths:
    raw_json: Path = data_dirpath / "arxiv_metadata.json"
    id_labels_pkl: Path = data_dirpath / "arxiv_id_labels.pkl"


@dataclass(frozen=True)
class MergedDataPaths:
    documents_labels_chunk_stem: Path = data_dirpath / "documents_labels_chunks"
    documents_labels_pkl: Path = data_dirpath / "documents_labels.pkl"
    documents_authors_labels_pkl: Path = data_dirpath / "documents_authors_labels.pkl"
    documents_authors_labels_citations_chunks_stem: Path = (
        data_dirpath / "documents_authors_labels_citations_chunks"
    )
    documents_authors_labels_citations_pkl: Path = (
        data_dirpath / "documents_authors_labels_citations.pkl"
    )
    documents_authors_labels_citations_most_cited_pkl: Path = (
        data_dirpath / "documents_authors_labels_citations_most_cited.pkl"
    )
    most_cited_subset_size: int = 10_000


@dataclass(frozen=True)
class DataPaths:
    d3: D3 = D3()
    arxiv: ArxivDataPaths = ArxivDataPaths()
    merged: MergedDataPaths = MergedDataPaths()


@dataclass(frozen=True)
class ModelVersions:
    spacy: str = "en_core_web_sm"
    word2vec: str = "word2vec-google-news-300"
    fasttext: str = "cc.en.300.bin"
    bert: str = "bert-base-uncased"
    scibert: str = "allenai/scibert_scivocab_uncased"


@dataclass
class ModelPaths:
    """File paths to pretrained models"""

    word2vec: Path = models_dirpath / ModelVersions.word2vec
    fasttext: Path = models_dirpath / ModelVersions.fasttext


@dataclass(frozen=True)
class CitationModelsResultsPaths:
    bibliographic_coupling_scores_most_cited_pkl: Path = (
        results_dirpath / "bibliographic_coupling_scores_most_cited.pkl"
    )
    co_citation_analysis_scores_most_cited_pkl: Path = (
        results_dirpath / "co_citation_analysis_scores_most_cited.pkl"
    )


@dataclass(frozen=True)
class LanguageModelsResultsPaths:
    spacy_tokenized_abstracts_most_cited_pkl: Path = (
        results_dirpath / "spacy_tokenized_abstracts_most_cited.pkl"
    )
    tfidf_embeddings_mapping_most_cited_pkl: Path = (
        results_dirpath / "tfidf_embeddings_most_cited.pkl"
    )
    tfidf_cosine_similarities_most_cited_pkl: Path = (
        results_dirpath / "tfidf_cosine_similarities_most_cited.pkl"
    )
    bm25_embeddings_mapping_most_cited_pkl: Path = (
        results_dirpath / "bm25_embeddings_most_cited.pkl"
    )
    bm25_cosine_similarities_most_cited_pkl: Path = (
        results_dirpath / "bm25_cosine_similarities_most_cited.pkl"
    )
    word2vec_embeddings_mapping_most_cited_pkl: Path = (
        results_dirpath / "word2vec_embeddings_most_cited.pkl"
    )
    word2vec_cosine_similarities_most_cited_pkl: Path = (
        results_dirpath / "word2vec_cosine_similarities_most_cited.pkl"
    )
    fasttext_embeddings_mapping_most_cited_pkl: Path = (
        results_dirpath / "fasttext_embeddings_most_cited.pkl"
    )
    fasttext_cosine_similarities_most_cited_pkl: Path = (
        results_dirpath / "fasttext_cosine_similarities_most_cited.pkl"
    )
    bert_tokenized_abstracts_most_cited_pt: Path = (
        results_dirpath / "bert_tokenized_abstracts_most_cited.pt"
    )
    bert_embeddings_mapping_most_cited_pkl: Path = (
        results_dirpath / "bert_embeddings_most_cited.pkl"
    )
    bert_cosine_similarities_most_cited_pkl: Path = (
        results_dirpath / "bert_cosine_similarities_most_cited.pkl"
    )
    scibert_tokenized_abstracts_most_cited_pt: Path = (
        results_dirpath / "scibert_tokenized_abstracts_most_cited.pt"
    )
    scibert_embeddings_mapping_most_cited_pkl: Path = (
        results_dirpath / "scibert_embeddings_most_cited.pkl"
    )
    scibert_cosine_similarities_most_cited_pkl: Path = (
        results_dirpath / "scibert_cosine_similarities_most_cited.pkl"
    )


@dataclass(frozen=True)
class ResultsPaths:
    citation_models: CitationModelsResultsPaths = CitationModelsResultsPaths()
    language_models: LanguageModelsResultsPaths = LanguageModelsResultsPaths()
