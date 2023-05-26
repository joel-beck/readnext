from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def root_path() -> Path:
    """Return project root path when pytest is executed from the project root directory."""
    return Path().cwd()


@pytest.fixture(scope="session")
def test_data_size() -> int:
    return 100
