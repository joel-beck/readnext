"""
Set file paths for reading and writing model results. Requires a .env file in the root
directory with the following variables:
- DATA_DIRPATH (default: `results`): path to the results directory where all model
results are stored
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

results_dirpath = Path(os.getenv("RESULTS_DIRPATH") or "results")


@dataclass(frozen=True)
class CitationModelsResultsPaths:
    bibliographic_coupling_most_cited_pkl: Path = (
        results_dirpath / "bibliographic_coupling_most_cited.pkl"
    )
    co_citation_analysis_most_cited_pkl: Path = (
        results_dirpath / "co_citation_analysis_most_cited.pkl"
    )


@dataclass(frozen=True)
class ResultsPaths:
    citation_models: CitationModelsResultsPaths = CitationModelsResultsPaths()
