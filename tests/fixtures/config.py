from pathlib import Path

import pytest

from readnext.config import PROJECT_PATH, MagicNumbers


@pytest.fixture(scope="session")
def root_path() -> Path:
    """Return project root path when pytest is executed from the project root directory."""
    return PROJECT_PATH


@pytest.fixture(scope="session")
def test_data_size() -> int:
    return MagicNumbers.unit_testing_size
