import polars as pl
import pytest

from readnext.utils.aliases import EmbeddingsFrame, Tokens


@pytest.fixture
def toy_citation_models_features_frame() -> pl.DataFrame:
    data = {
        "publication_date": ["2020-01-01", "2019-01-01", None, "2018-01-01", "2017-01-01"],
        "citationcount_document": [50, 100, 75, 50, 100],
        "citationcount_author": [1000, 2000, 3000, 1000, 2000],
    }
    return pl.DataFrame(data)


@pytest.fixture(scope="session")
def toy_document_tokens() -> Tokens:
    return ["a", "b", "c", "a", "b", "c", "d", "d", "d"]


@pytest.fixture(scope="session")
def toy_document_corpus() -> list[Tokens]:
    return [
        ["a", "b", "c", "d", "d", "d"],
        ["a", "b", "b", "c", "c", "c", "d"],
        ["a", "a", "a", "b", "c", "d"],
    ]


@pytest.fixture
def toy_embeddings_frame() -> EmbeddingsFrame:
    return pl.DataFrame(
        {
            "d3_document_id": [1, 2, 3],
            "embedding": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
        }
    )
