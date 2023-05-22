import pytest


@pytest.fixture
def semanticscholar_url() -> str:
    return "https://www.semanticscholar.org/paper/5cc2cfb77c9643760f4e7e18"


@pytest.fixture
def semanticscholar_id() -> str:
    return "5cc2cfb77c9643760f4e7e18"


@pytest.fixture
def arxiv_url() -> str:
    return "https://arxiv.org/abs/2101.10000"


@pytest.fixture
def arxiv_id() -> str:
    return "2101.10000"
