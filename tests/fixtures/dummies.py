import pytest

from readnext.modeling import DocumentInfo


@pytest.fixture
def dummy_document_info() -> DocumentInfo:
    return DocumentInfo(
        d3_document_id=1,
        title="Sample Paper",
        author="John Doe",
        publication_date="2000-01-01",
        arxiv_labels=["cs.AI", "cs.CL"],
        semanticscholar_url=(
            "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
        ),
        arxiv_url="https://arxiv.org/abs/2106.01572",
        abstract="This is a sample paper.",
    )
