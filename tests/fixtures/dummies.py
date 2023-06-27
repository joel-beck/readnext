import polars as pl
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, Features, Labels, Points, Ranks, Recommendations
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


@pytest.fixture
def dummy_document_identifier() -> DocumentIdentifier:
    return DocumentIdentifier(
        d3_document_id=1,
        semanticscholar_id="",
        semanticscholar_url="",
        arxiv_id="",
        arxiv_url="",
    )


@pytest.fixture
def dummy_features() -> Features:
    return Features(
        publication_date=pl.DataFrame(),
        citationcount_document=pl.DataFrame(),
        citationcount_author=pl.DataFrame(),
        co_citation_analysis=pl.DataFrame(),
        bibliographic_coupling=pl.DataFrame(),
        cosine_similarity=pl.DataFrame(),
        feature_weights=FeatureWeights(),
    )


@pytest.fixture
def dummy_ranks() -> Ranks:
    return Ranks(
        publication_date=pl.DataFrame(),
        citationcount_document=pl.DataFrame(),
        citationcount_author=pl.DataFrame(),
        co_citation_analysis=pl.DataFrame(),
        bibliographic_coupling=pl.DataFrame(),
    )


@pytest.fixture
def dummy_points() -> Points:
    return Points(
        publication_date=pl.DataFrame(),
        citationcount_document=pl.DataFrame(),
        citationcount_author=pl.DataFrame(),
        co_citation_analysis=pl.DataFrame(),
        bibliographic_coupling=pl.DataFrame(),
    )


@pytest.fixture
def dummy_labels() -> Labels:
    return Labels(arxiv=pl.DataFrame(), integer=pl.DataFrame())


@pytest.fixture
def dummy_recommendations() -> Recommendations:
    return Recommendations(
        citation_to_language=pl.DataFrame(),
        citation_to_language_candidates=pl.DataFrame(),
        language_to_citation=pl.DataFrame(),
        language_to_citation_candidates=pl.DataFrame(),
    )
