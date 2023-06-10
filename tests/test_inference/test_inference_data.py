import numpy as np
import polars as pl
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, InferenceData

# These imports must not come from `readnext.inference`, otherwise they are really
# imported twice with different module scopes and `isinstance()` checks fail.
from readnext.inference.constructor import Features, Labels, Ranks, Recommendations
from readnext.modeling import DocumentInfo

# SECTION: Document Identifier
seen_document_identifier_fixtures_slow_skip_ci = [
    "inference_data_seen_document_identifier",
    "inference_data_constructor_seen_document_identifier",
]
unseen_document_identifier_fixtures_skip_ci = [
    "inference_data_unseen_document_identifier",
    "inference_data_constructor_unseen_document_identifier",
]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "document_identifier", lazy_fixture(seen_document_identifier_fixtures_slow_skip_ci)
)
def test_inference_data_seen_document_identifier(
    document_identifier: DocumentIdentifier,
) -> None:
    assert isinstance(document_identifier, DocumentIdentifier)

    assert isinstance(document_identifier.semanticscholar_url, str)
    assert (
        document_identifier.semanticscholar_url
        == "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )

    assert isinstance(document_identifier.semanticscholar_id, str)
    assert document_identifier.semanticscholar_id == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    assert isinstance(document_identifier.arxiv_url, str)
    assert document_identifier.arxiv_url == "https://arxiv.org/abs/1706.03762"

    assert isinstance(document_identifier.arxiv_id, str)
    assert document_identifier.arxiv_id == "1706.03762"

    assert isinstance(document_identifier.d3_document_id, int)
    assert document_identifier.d3_document_id == 13756489


@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "document_identifier", lazy_fixture(unseen_document_identifier_fixtures_skip_ci)
)
def test_inference_data_unseen_document_identifier(
    document_identifier: DocumentIdentifier,
) -> None:
    assert isinstance(document_identifier, DocumentIdentifier)

    assert isinstance(document_identifier.semanticscholar_url, str)
    assert (
        document_identifier.semanticscholar_url
        == "https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"
    )

    assert isinstance(document_identifier.semanticscholar_id, str)
    assert document_identifier.semanticscholar_id == "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    assert isinstance(document_identifier.arxiv_url, str)
    assert document_identifier.arxiv_url == "https://arxiv.org/abs/2303.08774"

    assert isinstance(document_identifier.arxiv_id, str)
    assert document_identifier.arxiv_id == "2303.08774"

    assert isinstance(document_identifier.d3_document_id, int)
    assert document_identifier.d3_document_id == -1


# SECTION: Document Info
seen_document_info_fixtures_slow_skip_ci = [
    "inference_data_seen_document_info",
    "inference_data_constructor_seen_document_info",
]
unseen_document_info_fixtures_skip_ci = [
    "inference_data_unseen_document_info",
    "inference_data_constructor_unseen_document_info",
]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("document_info", lazy_fixture(seen_document_info_fixtures_slow_skip_ci))
def test_inference_data_seen_document_info(document_info: DocumentInfo) -> None:
    assert isinstance(document_info, DocumentInfo)

    assert isinstance(document_info.d3_document_id, int)
    assert document_info.d3_document_id == 13756489

    assert isinstance(document_info.title, str)
    assert document_info.title == "Attention is All you Need"

    assert isinstance(document_info.author, str)
    assert document_info.author == "Lukasz Kaiser"

    assert isinstance(document_info.abstract, str)
    assert len(document_info.abstract) > 0

    assert isinstance(document_info.arxiv_labels, list)
    assert all(isinstance(label, str) for label in document_info.arxiv_labels)
    assert document_info.arxiv_labels == ["cs.CL", "cs.LG"]


@pytest.mark.skip_ci
@pytest.mark.parametrize("document_info", lazy_fixture(unseen_document_info_fixtures_skip_ci))
def test_inference_data_unseen_document_info(document_info: DocumentInfo) -> None:
    assert isinstance(document_info, DocumentInfo)

    assert isinstance(document_info.d3_document_id, int)
    assert document_info.d3_document_id == -1

    assert isinstance(document_info.title, str)
    assert document_info.title == "GPT-4 Technical Report"

    assert isinstance(document_info.author, str)
    assert document_info.author == ""

    assert isinstance(document_info.abstract, str)
    assert len(document_info.abstract) > 0

    assert isinstance(document_info.arxiv_labels, list)
    assert document_info.arxiv_labels == []


# SECTION: Features
feature_fixtures_slow_skip_ci = [
    "inference_data_seen_features",
    "inference_data_constructor_seen_features",
]

feature_fixtures_skip_ci = [
    "inference_data_unseen_features",
    "inference_data_constructor_unseen_features",
]


@pytest.mark.parametrize(
    "features",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in feature_fixtures_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in feature_fixtures_skip_ci
        ],
    ],
)
def test_inference_data_features(features: Features) -> None:
    assert isinstance(features, Features)

    assert isinstance(features.publication_date, pl.DataFrame)
    assert features.publication_date.name == "publication_date"
    assert features.publication_date.dtype == pl.Utf8
    assert features.publication_date.index.name == "document_id"
    assert features.publication_date.index.dtype == pl.Int64Dtype()

    assert isinstance(features.citationcount_document, pl.DataFrame)
    assert features.citationcount_document.name == "citationcount_document"
    assert features.citationcount_document.dtype == pl.Int64Dtype()
    assert features.citationcount_document.index.name == "document_id"
    assert features.citationcount_document.index.dtype == pl.Int64Dtype()

    assert isinstance(features.citationcount_author, pl.DataFrame)
    assert features.citationcount_author.name == "citationcount_author"
    assert features.citationcount_author.dtype == pl.Int64Dtype()
    assert features.citationcount_author.index.name == "document_id"
    assert features.citationcount_author.index.dtype == pl.Int64Dtype()

    assert isinstance(features.co_citation_analysis, pl.DataFrame)
    assert features.co_citation_analysis.name == "co_citation_analysis"
    assert features.co_citation_analysis.dtype == np.dtype("int64")
    assert features.co_citation_analysis.index.name == "document_id"
    assert features.co_citation_analysis.index.dtype == pl.Int64Dtype()

    assert isinstance(features.bibliographic_coupling, pl.DataFrame)
    assert features.bibliographic_coupling.name == "bibliographic_coupling"
    assert features.bibliographic_coupling.dtype == np.dtype("int64")
    assert features.bibliographic_coupling.index.name == "document_id"
    assert features.bibliographic_coupling.index.dtype == pl.Int64Dtype()

    assert isinstance(features.cosine_similarity, pl.DataFrame)
    assert features.cosine_similarity.name == "cosine_similarity"
    assert features.cosine_similarity.dtype == np.dtype("float64")
    assert features.cosine_similarity.index.name == "document_id"
    assert features.cosine_similarity.index.dtype == pl.Int64Dtype()

    assert isinstance(features.feature_weights, FeatureWeights)


# SECTION: Ranks
rank_fixtures_slow_skip_ci = [
    "inference_data_seen_ranks",
    "inference_data_constructor_seen_ranks",
]

rank_fixtures_skip_ci = [
    "inference_data_unseen_ranks",
    "inference_data_constructor_unseen_ranks",
]


@pytest.mark.parametrize(
    "ranks",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in rank_fixtures_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in rank_fixtures_skip_ci
        ],
    ],
)
def test_inference_data_ranks(ranks: Ranks) -> None:
    assert isinstance(ranks, Ranks)

    assert isinstance(ranks.publication_date, pl.DataFrame)
    assert ranks.publication_date.name == "publication_date_rank"
    assert ranks.publication_date.dtype == np.dtype("float64")
    assert ranks.publication_date.index.name == "document_id"
    assert ranks.publication_date.index.dtype == pl.Int64Dtype()

    assert isinstance(ranks.citationcount_document, pl.DataFrame)
    assert ranks.citationcount_document.name == "citationcount_document_rank"
    assert ranks.citationcount_document.dtype == np.dtype("float64")
    assert ranks.citationcount_document.index.name == "document_id"
    assert ranks.citationcount_document.index.dtype == pl.Int64Dtype()

    assert isinstance(ranks.citationcount_author, pl.DataFrame)
    assert ranks.citationcount_author.name == "citationcount_author_rank"
    assert ranks.citationcount_author.dtype == np.dtype("float64")
    assert ranks.citationcount_author.index.name == "document_id"
    assert ranks.citationcount_author.index.dtype == pl.Int64Dtype()

    assert isinstance(ranks.co_citation_analysis, pl.DataFrame)
    assert ranks.co_citation_analysis.name == "co_citation_analysis_rank"
    assert ranks.co_citation_analysis.dtype == np.dtype("float64")
    assert ranks.co_citation_analysis.index.name == "document_id"
    assert ranks.co_citation_analysis.index.dtype == pl.Int64Dtype()

    assert isinstance(ranks.bibliographic_coupling, pl.DataFrame)
    assert ranks.bibliographic_coupling.name == "bibliographic_coupling_rank"
    assert ranks.bibliographic_coupling.dtype == np.dtype("float64")
    assert ranks.bibliographic_coupling.index.name == "document_id"
    assert ranks.bibliographic_coupling.index.dtype == pl.Int64Dtype()

    assert isinstance(ranks.cosine_similarity, pl.DataFrame)
    assert ranks.cosine_similarity.name == "cosine_similarity_rank"
    assert ranks.cosine_similarity.dtype == np.dtype("float64")
    assert ranks.cosine_similarity.index.name == "document_id"
    assert ranks.cosine_similarity.index.dtype == np.dtype("int64")


# SECTION: Labels
label_fixtures_slow_skip_ci = [
    "inference_data_seen_labels",
    "inference_data_constructor_seen_labels",
]

label_fixtures_skip_ci = [
    "inference_data_unseen_labels",
    "inference_data_constructor_unseen_labels",
]


@pytest.mark.parametrize(
    "labels",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in label_fixtures_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in label_fixtures_skip_ci
        ],
    ],
)
def test_inference_data_labels(labels: Labels) -> None:
    assert isinstance(labels, Labels)

    assert isinstance(labels.arxiv, pl.DataFrame)
    assert labels.arxiv.name == "arxiv_labels"
    assert labels.arxiv.dtype == object
    assert labels.arxiv.index.name == "document_id"
    assert labels.arxiv.index.dtype == pl.Int64Dtype()
    assert all(isinstance(labels, list) for labels in labels.arxiv)
    assert all(isinstance(label, str) for labels in labels.arxiv for label in labels)
    assert all(len(labels) > 0 for labels in labels.arxiv)

    assert isinstance(labels.integer, pl.DataFrame)
    assert labels.integer.name == "integer_labels"
    assert labels.integer.dtype == np.dtype("int64")
    assert labels.integer.index.name == "document_id"
    assert labels.integer.index.dtype == pl.Int64Dtype()


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("labels", lazy_fixture(label_fixtures_slow_skip_ci))
def test_inference_data_seen_labels(labels: Labels) -> None:
    assert labels.integer.unique().tolist() == [0, 1]


@pytest.mark.skip_ci
@pytest.mark.parametrize("labels", lazy_fixture(label_fixtures_skip_ci))
def test_inference_data_unseen_labels(labels: Labels) -> None:
    # for unseen papers no arxiv labels are available, thus there is no intersection
    # with the candidate paper arxiv labels
    assert labels.integer.unique().tolist() == [0]


# SECTION: Recommendations
recommendations_fixtures_slow_skip_ci = [
    "inference_data_seen_recommendations",
    "inference_data_constructor_seen_recommendations",
]
recommendations_fixtures_skip_ci = [
    "inference_data_unseen_recommendations",
    "inference_data_constructor_unseen_recommendations",
]


@pytest.mark.parametrize(
    "recommendations",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in recommendations_fixtures_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in recommendations_fixtures_skip_ci
        ],
    ],
)
def test_inference_data_recommendations(recommendations: Recommendations) -> None:
    assert isinstance(recommendations, Recommendations)


recommendation_dataframe_fixtures_citation_to_language_candidates_slow_skip_ci = [
    "inference_data_seen_recommendations_citation_to_language_candidates",
]
recommendation_dataframe_fixtures_citation_to_language_candidates_skip_ci = [
    "inference_data_unseen_recommendations_citation_to_language_candidates",
]

recommendation_dataframe_fixtures_citation_to_language_slow_skip_ci = [
    "inference_data_seen_recommendations_citation_to_language",
]
recommendation_dataframe_fixtures_citation_to_language_skip_ci = [
    "inference_data_unseen_recommendations_citation_to_language",
]

recommendation_dataframe_fixtures_language_to_citation_candidates_slow_skip_ci = [
    "inference_data_seen_recommendations_language_to_citation_candidates",
]
recommendation_dataframe_fixtures_language_to_citation_candidates_skip_ci = [
    "inference_data_unseen_recommendations_language_to_citation_candidates",
]

recommendation_dataframe_fixtures_language_to_citation_slow_skip_ci = [
    "inference_data_seen_recommendations_language_to_citation",
]
recommendation_dataframe_fixtures_language_to_citation_skip_ci = [
    "inference_data_unseen_recommendations_language_to_citation",
]

recommendation_dataframe_fixtures_citation_features_slow_skip_ci = (
    recommendation_dataframe_fixtures_citation_to_language_candidates_slow_skip_ci
    + recommendation_dataframe_fixtures_language_to_citation_slow_skip_ci
)
recommendation_dataframe_fixtures_citation_features_skip_ci = (
    recommendation_dataframe_fixtures_citation_to_language_candidates_skip_ci
    + recommendation_dataframe_fixtures_language_to_citation_skip_ci
)

recommendation_dataframe_fixtures_language_features_slow_skip_ci = (
    recommendation_dataframe_fixtures_language_to_citation_candidates_slow_skip_ci
    + recommendation_dataframe_fixtures_citation_to_language_slow_skip_ci
)
recommendation_dataframe_fixtures_language_features_skip_ci = (
    recommendation_dataframe_fixtures_language_to_citation_candidates_skip_ci
    + recommendation_dataframe_fixtures_citation_to_language_skip_ci
)

recommendation_dataframe_fixtures_slow_skip_ci = (
    recommendation_dataframe_fixtures_citation_features_slow_skip_ci
    + recommendation_dataframe_fixtures_language_features_slow_skip_ci
)
recommendation_dataframe_fixtures_skip_ci = (
    recommendation_dataframe_fixtures_citation_features_skip_ci
    + recommendation_dataframe_fixtures_language_features_skip_ci
)


@pytest.mark.parametrize(
    "recommendations_dataframe",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in recommendation_dataframe_fixtures_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in recommendation_dataframe_fixtures_skip_ci
        ],
    ],
)
def test_inference_data_recommendations_dataframes(recommendations_dataframe: pl.DataFrame) -> None:
    assert isinstance(recommendations_dataframe, pl.DataFrame)

    assert recommendations_dataframe.index.name == "document_id"
    assert recommendations_dataframe.index.dtype == pl.Int64Dtype()


@pytest.mark.parametrize(
    "recommendations_dataframe",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in recommendation_dataframe_fixtures_citation_features_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in recommendation_dataframe_fixtures_citation_features_skip_ci
        ],
    ],
)
def test_inference_data_recommendations_dataframes_citation_features(
    recommendations_dataframe: pl.DataFrame,
) -> None:
    assert recommendations_dataframe.shape[1] == 9
    assert recommendations_dataframe.columns.tolist() == [
        "weighted_rank",
        "title",
        "author",
        "arxiv_labels",
        "publication_date",
        "citationcount_document",
        "citationcount_author",
        "co_citation_analysis",
        "bibliographic_coupling",
    ]
    assert recommendations_dataframe.dtypes.tolist() == [
        np.dtype("float64"),
        pl.StringDtype(),
        pl.StringDtype(),
        object,
        pl.StringDtype(),
        pl.Int64Dtype(),
        pl.Int64Dtype(),
        np.dtype("int64"),
        np.dtype("int64"),
    ]


@pytest.mark.parametrize(
    "recommendations_dataframe",
    [
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.slow, pytest.mark.skip_ci))
            for fixture in recommendation_dataframe_fixtures_language_features_slow_skip_ci
        ],
        *[
            pytest.param(lazy_fixture(fixture), marks=(pytest.mark.skip_ci))
            for fixture in recommendation_dataframe_fixtures_language_features_skip_ci
        ],
    ],
)
def test_inference_data_recommendations_dataframes_language_features(
    recommendations_dataframe: pl.DataFrame,
) -> None:
    assert recommendations_dataframe.shape[1] == 4
    assert recommendations_dataframe.columns.tolist() == [
        "cosine_similarity",
        "title",
        "author",
        "arxiv_labels",
    ]
    assert recommendations_dataframe.dtypes.tolist() == [
        np.dtype("float64"),
        pl.StringDtype(),
        pl.StringDtype(),
        object,
    ]
    # check that cosine similarities are between 0 and 1
    assert recommendations_dataframe["cosine_similarity"].min() >= 0
    assert recommendations_dataframe["cosine_similarity"].max() <= 1

    # check that the cosine similarities are sorted in descending order
    assert all(recommendations_dataframe["cosine_similarity"].diff().dropna().to_numpy() <= 0)  # type: ignore # noqa: E501


def test_kw_only_initialization_inference_data() -> None:
    with pytest.raises(TypeError):
        InferenceData(
            DocumentIdentifier(  # type: ignore
                d3_document_id=-1,
                semanticscholar_id="8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
                semanticscholar_url="https://www.semanticscholar.org/paper/8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48",
                arxiv_id="2303.08774",
                arxiv_url="https://arxiv.org/abs/2303.08774",
            ),
            DocumentInfo(
                d3_document_id=-1,
                title="Title",
                author="Author",
                abstract="Abstract",
                arxiv_labels=[],
            ),
            Features(
                publication_date=pl.DataFrame(),
                citationcount_document=pl.DataFrame(),
                citationcount_author=pl.DataFrame(),
                co_citation_analysis=pl.DataFrame(),
                bibliographic_coupling=pl.DataFrame(),
                cosine_similarity=pl.DataFrame(),
                feature_weights=FeatureWeights(),
            ),
            Ranks(
                publication_date=pl.DataFrame(),
                citationcount_document=pl.DataFrame(),
                citationcount_author=pl.DataFrame(),
                co_citation_analysis=pl.DataFrame(),
                bibliographic_coupling=pl.DataFrame(),
            ),
            Labels(
                arxiv=pl.DataFrame(),
                integer=pl.DataFrame(),
            ),
            Recommendations(
                citation_to_language_candidates=pl.DataFrame(),
                citation_to_language=pl.DataFrame(),
                language_to_citation_candidates=pl.DataFrame(),
                language_to_citation=pl.DataFrame(),
            ),
        )
