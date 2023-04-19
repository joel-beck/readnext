from pathlib import Path

import pytest


@pytest.fixture
def root_path() -> Path:
    return Path(__file__).parent.parent
