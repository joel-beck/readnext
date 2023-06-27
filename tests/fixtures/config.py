from pathlib import Path

import pytest

from readnext.config import MagicNumbers, PROJECT_PATH


@pytest.fixture(scope="session")
def root_path() -> Path:
    """Return project root path when pytest is executed from the project root directory."""
    return PROJECT_PATH


@pytest.fixture(scope="session")
def test_data_size() -> int:
    return MagicNumbers.documents_frame_test_size
