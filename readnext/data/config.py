"""
Set file paths for reading and writing data. Requires a .env file in the root directory
with the following variables:
- DATA_DIRPATH (required): path to the data directory where all data files are stored
- DOCUMENTS_METADATA_FILENAME (optional): name of the file containing the downloaded D3
documents in jsonl format
- AUTHORS_METADATA_FILENAME (optional): name of the file containing the downloaded D3
authors in jsonl format
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# os.getenv() returns str | None, os.environ[] always returns str and raises KeyError if
# environment variable is not set
data_dirpath = Path(os.getenv("DATA_DIRPATH") or "data")

documents_metadata_json_filename = (
    os.getenv("DOCUMENTS_METADATA_FILENAME") or "2022-11-30-papers.jsonl"
)

authors_metadata_json_filename = (
    os.getenv("AUTHORS_METADATA_FILENAME") or "2022-11-30-authors.jsonl"
)


@dataclass
class D3DocumentsDataPaths:
    raw_json: Path = data_dirpath / documents_metadata_json_filename
    chunks_stem: Path = data_dirpath / "2022-11-30_documents_chunks"
    full_pkl: Path = data_dirpath / "2022-11-30_documents_full.pkl"
    preprocessed_chunks_stem: Path = data_dirpath / "documents_preprocessed_chunks"


@dataclass
class D3AuthorsDataPaths:
    raw_json: Path = data_dirpath / authors_metadata_json_filename
    most_cited_pkl: Path = data_dirpath / "2022-11-30_authors_most_cited.pkl"
    full_pkl: Path = data_dirpath / "2022-11-30_authors_full.pkl"


@dataclass
class D3:
    documents: D3DocumentsDataPaths = D3DocumentsDataPaths()
    authors: D3AuthorsDataPaths = D3AuthorsDataPaths()


@dataclass
class ArxivDataPaths:
    raw_json: Path = data_dirpath / "arxiv_metadata.json"
    id_labels_pkl: Path = data_dirpath / "arxiv_id_labels.pkl"


@dataclass
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
    most_cited_subset_size: int = 1_000


@dataclass
class DataPaths:
    d3: D3 = D3()
    arxiv: ArxivDataPaths = ArxivDataPaths()
    merged: MergedDataPaths = MergedDataPaths()
