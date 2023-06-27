"""
Project configuration file.

- Sets file paths for reading and writing data
- Sets file paths for reading pretrained models
- Sets file paths for reading and writing model results
- Specifies Model Versions

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
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = Path(__file__).resolve().parent.parent

data_dirpath = Path(os.getenv("DATA_DIRPATH", "data"))
models_dirpath = Path(os.getenv("MODELS_DIRPATH", "models"))
results_dirpath = Path(os.getenv("RESULTS_DIRPATH", "results"))

documents_metadata_json_filename = os.getenv(
    "DOCUMENTS_METADATA_FILENAME", "2022-11-30_papers.jsonl"
)
authors_metadata_json_filename = os.getenv("AUTHORS_METADATA_FILENAME", "2022-11-30_authors.jsonl")
arxiv_metadata_json_filename = os.getenv("ARXIV_METADATA_FILENAME", "arxiv_metadata.json")


@dataclass(frozen=True)
class RawDataPaths:
    """
    Sets file paths for the raw documents, authors and arxiv metadata in JSON and
    Parquet format.
    """

    documents_json: Path = data_dirpath / documents_metadata_json_filename
    documents_parquet: Path = data_dirpath / "2022-11-30_documents.parquet"
    authors_json: Path = data_dirpath / authors_metadata_json_filename
    authors_parquet: Path = data_dirpath / "2022-11-30_authors.parquet"
    arxiv_labels_json: Path = data_dirpath / arxiv_metadata_json_filename
    arxiv_labels_parquet: Path = data_dirpath / "arxiv_metadata.parquet"


@dataclass(frozen=True)
class MergedDataPaths:
    """
    Sets file paths for all processed data files after adding information to the raw
    documents data.
    """

    documents_labels: Path = data_dirpath / "documents_labels.parquet"
    documents_authors_labels: Path = data_dirpath / "documents_authors_labels.parquet"
    documents_authors_labels_citations: Path = (
        data_dirpath / "documents_authors_labels_citations.parquet"
    )
    documents_frame: Path = data_dirpath / "documents_frame.parquet"


@dataclass(frozen=True)
class DataPaths:
    """Collects file paths for all data files."""

    raw: RawDataPaths = RawDataPaths()  # noqa: RUF009
    merged: MergedDataPaths = MergedDataPaths()  # noqa: RUF009


@dataclass(frozen=True)
class ModelVersions:
    """Specifies the versions of pretrained models."""

    spacy: str = "en_core_web_sm"
    word2vec: str = "word2vec-google-news-300"
    glove: str = "glove.6B.300d"
    fasttext: str = "cc.en.300.bin"
    bert: str = "bert-base-uncased"
    scibert: str = "allenai/scibert_scivocab_uncased"
    longformer: str = "allenai/longformer-base-4096"


@dataclass
class ModelPaths:
    """Sets file paths to pretrained models."""

    word2vec: Path = models_dirpath / ModelVersions.word2vec
    glove: Path = models_dirpath / f"{ModelVersions.glove}.txt"
    fasttext: Path = models_dirpath / ModelVersions.fasttext


@dataclass(frozen=True)
class CitationModelsResultsPaths:
    """Sets file paths for citation model results."""

    bibliographic_coupling_scores_parquet: Path = (
        results_dirpath / "bibliographic_coupling_scores.parquet"
    )
    co_citation_analysis_scores_parquet: Path = (
        results_dirpath / "co_citation_analysis_scores.parquet"
    )


@dataclass(frozen=True)
class LanguageModelsResultsPaths:
    """Sets file paths for language model results."""

    spacy_tokens_frame_parquet: Path = results_dirpath / "spacy_tokens_frame.parquet"

    tfidf_embeddings_frame_parquet: Path = results_dirpath / "tfidf_embeddings_frame.parquet"
    tfidf_cosine_similarities_parquet: Path = results_dirpath / "tfidf_cosine_similarities.parquet"

    bm25_embeddings_frame_parquet: Path = results_dirpath / "bm25_embeddings_frame.parquet"
    bm25_cosine_similarities_parquet: Path = results_dirpath / "bm25_cosine_similarities.parquet"

    word2vec_embeddings_frame_parquet: Path = results_dirpath / "word2vec_embeddings_frame.parquet"
    word2vec_cosine_similarities_parquet: Path = (
        results_dirpath / "word2vec_cosine_similarities.parquet"
    )

    glove_embeddings_frame_parquet: Path = results_dirpath / "glove_embeddings_frame.parquet"
    glove_cosine_similarities_parquet: Path = results_dirpath / "glove_cosine_similarities.parquet"

    fasttext_embeddings_frame_parquet: Path = results_dirpath / "fasttext_embeddings_frame.parquet"
    fasttext_cosine_similarities_parquet: Path = (
        results_dirpath / "fasttext_cosine_similarities.parquet"
    )

    bert_token_ids_frame_parquet: Path = results_dirpath / "bert_token_ids_frame.parquet"
    bert_embeddings_frame_parquet: Path = results_dirpath / "bert_embeddings_frame.parquet"
    bert_cosine_similarities_parquet: Path = results_dirpath / "bert_cosine_similarities.parquet"

    scibert_token_ids_frame_parquet: Path = results_dirpath / "scibert_token_ids_frame.parquet"
    scibert_embeddings_frame_parquet: Path = results_dirpath / "scibert_embeddings_frame.parquet"
    scibert_cosine_similarities_parquet: Path = (
        results_dirpath / "scibert_cosine_similarities.parquet"
    )

    longformer_token_ids_frame_parquet: Path = (
        results_dirpath / "longformer_token_ids_frame.parquet"
    )
    longformer_embeddings_frame_parquet: Path = (
        results_dirpath / "longformer_embeddings_frame.parquet"
    )
    longformer_cosine_similarities_parquet: Path = (
        results_dirpath / "longformer_cosine_similarities.parquet"
    )


@dataclass(frozen=True)
class ResultsPaths:
    """Collects file paths for all result files."""

    citation_models: CitationModelsResultsPaths = CitationModelsResultsPaths()  # noqa: RUF009
    language_models: LanguageModelsResultsPaths = LanguageModelsResultsPaths()  # noqa: RUF009


@dataclass
class MagicNumbers:
    """
    Sets numeric values for dataset sizes, scoring and the candidate and final
    recommendation list.
    """

    documents_frame_intermediate_cutoff: int = 1_000_000
    documents_frame_final_size: int = 10_000
    documents_frame_test_size: int = 100
    scoring_limit: int = 100
    n_candidates: int = 20
    n_recommendations: int = 20


@dataclass(frozen=True)
class ColumnOrder:
    """
    Sets the column order of dataframe outputs.
    """

    documents_frame: list[str] = field(
        default_factory=lambda: [
            "d3_document_id",
            "d3_author_id",
            "title",
            "author",
            "publication_date",
            "publication_date_rank",
            "citationcount_document",
            "citationcount_document_rank",
            "citationcount_author",
            "citationcount_author_rank",
            "citations",
            "references",
            "abstract",
            "semanticscholar_id",
            "semanticscholar_url",
            "semanticscholar_tags",
            "arxiv_id",
            "arxiv_url",
            "arxiv_labels",
        ]
    )
    recommendations_citation: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "weighted_points",
            "title",
            "author",
            "arxiv_labels",
            "integer_label",
            "semanticscholar_url",
            "arxiv_url",
            "publication_date",
            "publication_date_points",
            "citationcount_document",
            "citationcount_document_points",
            "citationcount_author",
            "citationcount_author_points",
            "co_citation_analysis_score",
            "co_citation_analysis_points",
            "bibliographic_coupling_score",
            "bibliographic_coupling_points",
        ]
    )
    recommendations_language: list[str] = field(
        default_factory=lambda: [
            "candidate_d3_document_id",
            "cosine_similarity",
            "title",
            "author",
            "publication_date",
            "arxiv_labels",
            "integer_label",
            "semanticscholar_url",
            "arxiv_url",
        ]
    )
