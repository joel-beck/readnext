"""
- Specify Model Versions
- Set file paths for reading pretrained models
- Set file paths for reading and writing model results

Requires a .env file in the root directory with the following variables:
- MODELS_DIRPATH (default: `models`): path to the models directory where all pretrained
models are stored
- RESULTS_DIRPATH (default: `results`): path to the results directory where all model
results are stored
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

models_dirpath = Path(os.getenv("MODELS_DIRPATH") or "models")
results_dirpath = Path(os.getenv("RESULTS_DIRPATH") or "results")


@dataclass(frozen=True)
class ModelVersions:
    spacy: str = "en_core_web_sm"
    fasttext: str = "cc.en.300.bin"
    scibert: str = "allenai/scibert_scivocab_uncased"


@dataclass
class ModelPaths:
    """File paths to pretrained models"""

    fasttext: Path = models_dirpath / ModelVersions.fasttext


@dataclass(frozen=True)
class CitationModelsResultsPaths:
    bibliographic_coupling_most_cited_pkl: Path = (
        results_dirpath / "bibliographic_coupling_most_cited.pkl"
    )
    co_citation_analysis_most_cited_pkl: Path = (
        results_dirpath / "co_citation_analysis_most_cited.pkl"
    )


@dataclass(frozen=True)
class LanguageModelsResultsPaths:
    spacy_tokenized_abstracts_most_cited: Path = (
        results_dirpath / "spacy_tokenized_abstracts_most_cited.pkl"
    )


@dataclass(frozen=True)
class ResultsPaths:
    citation_models: CitationModelsResultsPaths = CitationModelsResultsPaths()
    language_models: LanguageModelsResultsPaths = LanguageModelsResultsPaths()
