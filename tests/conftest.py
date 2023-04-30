from pathlib import Path

import pytest


@pytest.fixture
def root_path() -> Path:
    """Return project root path when pytest is executed from the project root directory."""
    return Path().cwd()
