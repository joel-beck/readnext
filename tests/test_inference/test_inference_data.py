import numpy as np
import pandas as pd
import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, InferenceData

# These imports must not come from `readnext.inference`, otherwise they are really
# imported twice with different module scopes and `isinstance()` checks fail.
from readnext.inference.inference_data_constructor import Features, Labels, Ranks, Recommendations
from readnext.modeling import DocumentInfo

# SECTION: Document Identifier
seen_document_identifier_fixtures = [
    "inference_data_seen_document_identifier",
    "inference_data_constructor_seen_document_identifier",
]
unseen_document_identifier_fixtures = [
    "inference_data_unseen_document_identifier",
    "inference_data_constructor_unseen_document_identifier",
]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("document_identifier", lazy_fixture(seen_document_identifier_fixtures))
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


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("document_identifier", lazy_fixture(unseen_document_identifier_fixtures))
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
seen_document_info_fixtures = [
    "inference_data_seen_document_info",
    "inference_data_constructor_seen_document_info",
]
unseen_document_info_fixtures = [
    "inference_data_unseen_document_info",
    "inference_data_constructor_unseen_document_info",
]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("document_info", lazy_fixture(seen_document_info_fixtures))
def test_inference_data_seen_document_info(document_info: DocumentInfo) -> None:
    assert isinstance(document_info, DocumentInfo)

    assert isinstance(document_info.d3_document_id, int)
    assert document_info.d3_document_id == 13756489

    assert isinstance(document_info.title, str)
    assert document_info.title == "Attention is All you Need"

    assert isinstance(document_info.author, str)
    assert document_info.author == "Lukasz Kaiser"

    assert isinstance(document_info.abstract, str)
    # abstract is not set for seen documents
    # TODO: Should this be the case?
    assert len(document_info.abstract) == 0

    assert isinstance(document_info.arxiv_labels, list)
    assert all(isinstance(label, str) for label in document_info.arxiv_labels)
    assert document_info.arxiv_labels == ["cs.CL", "cs.LG"]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("document_info", lazy_fixture(unseen_document_info_fixtures))
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
feature_fixtures = [
    "inference_data_seen_features",
    "inference_data_constructor_seen_features",
    "inference_data_unseen_features",
    "inference_data_constructor_unseen_features",
]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("features", lazy_fixture(feature_fixtures))
def test_inference_data_features(features: Features) -> None:
    assert isinstance(features, Features)

    assert isinstance(features.publication_date, pd.Series)
    assert features.publication_date.name == "publication_date"
    assert features.publication_date.dtype == pd.StringDtype()
    assert features.publication_date.index.name == "document_id"
    assert features.publication_date.index.dtype == pd.Int64Dtype()

    assert isinstance(features.citationcount_document, pd.Series)
    assert features.citationcount_document.name == "citationcount_document"
    assert features.citationcount_document.dtype == pd.Int64Dtype()
    assert features.citationcount_document.index.name == "document_id"
    assert features.citationcount_document.index.dtype == pd.Int64Dtype()

    assert isinstance(features.citationcount_author, pd.Series)
    assert features.citationcount_author.name == "citationcount_author"
    assert features.citationcount_author.dtype == pd.Int64Dtype()
    assert features.citationcount_author.index.name == "document_id"
    assert features.citationcount_author.index.dtype == pd.Int64Dtype()

    assert isinstance(features.co_citation_analysis, pd.Series)
    assert features.co_citation_analysis.name == "co_citation_analysis"
    assert features.co_citation_analysis.dtype == np.dtype("int64")
    assert features.co_citation_analysis.index.name == "document_id"
    assert features.co_citation_analysis.index.dtype == pd.Int64Dtype()

    assert isinstance(features.bibliographic_coupling, pd.Series)
    assert features.bibliographic_coupling.name == "bibliographic_coupling"
    assert features.bibliographic_coupling.dtype == np.dtype("int64")
    assert features.bibliographic_coupling.index.name == "document_id"
    assert features.bibliographic_coupling.index.dtype == pd.Int64Dtype()

    assert isinstance(features.cosine_similarity, pd.Series)
    assert features.cosine_similarity.name == "cosine_similarity"
    assert features.cosine_similarity.dtype == np.dtype("float64")
    assert features.cosine_similarity.index.name == "document_id"
    assert features.cosine_similarity.index.dtype == pd.Int64Dtype()

    assert isinstance(features.feature_weights, FeatureWeights)


# SECTION: Ranks
ranks_fixtures = [
    "inference_data_seen_ranks",
    "inference_data_constructor_seen_ranks",
    "inference_data_unseen_ranks",
    "inference_data_constructor_unseen_ranks",
]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("ranks", lazy_fixture(ranks_fixtures))
def test_inference_data_ranks(ranks: Ranks) -> None:
    assert isinstance(ranks, Ranks)

    assert isinstance(ranks.publication_date, pd.Series)
    assert ranks.publication_date.name == "publication_date_rank"
    assert ranks.publication_date.dtype == np.dtype("float64")
    assert ranks.publication_date.index.name == "document_id"
    assert ranks.publication_date.index.dtype == pd.Int64Dtype()

    assert isinstance(ranks.citationcount_document, pd.Series)
    assert ranks.citationcount_document.name == "citationcount_document_rank"
    assert ranks.citationcount_document.dtype == np.dtype("float64")
    assert ranks.citationcount_document.index.name == "document_id"
    assert ranks.citationcount_document.index.dtype == pd.Int64Dtype()

    assert isinstance(ranks.citationcount_author, pd.Series)
    assert ranks.citationcount_author.name == "citationcount_author_rank"
    assert ranks.citationcount_author.dtype == np.dtype("float64")
    assert ranks.citationcount_author.index.name == "document_id"
    assert ranks.citationcount_author.index.dtype == pd.Int64Dtype()

    assert isinstance(ranks.co_citation_analysis, pd.Series)
    assert ranks.co_citation_analysis.name == "co_citation_analysis_rank"
    assert ranks.co_citation_analysis.dtype == np.dtype("float64")
    assert ranks.co_citation_analysis.index.name == "document_id"
    assert ranks.co_citation_analysis.index.dtype == pd.Int64Dtype()

    assert isinstance(ranks.bibliographic_coupling, pd.Series)
    assert ranks.bibliographic_coupling.name == "bibliographic_coupling_rank"
    assert ranks.bibliographic_coupling.dtype == np.dtype("float64")
    assert ranks.bibliographic_coupling.index.name == "document_id"
    assert ranks.bibliographic_coupling.index.dtype == pd.Int64Dtype()

    assert isinstance(ranks.cosine_similarity, pd.Series)
    assert ranks.cosine_similarity.name == "cosine_similarity_rank"
    assert ranks.cosine_similarity.dtype == np.dtype("float64")
    assert ranks.cosine_similarity.index.name == "document_id"
    assert ranks.cosine_similarity.index.dtype == np.dtype("int64")


# SECTION: Labels
seen_labels_fixtures = ["inference_data_seen_labels", "inference_data_constructor_seen_labels"]
unseen_labels_fixtures = [
    "inference_data_unseen_labels",
    "inference_data_constructor_unseen_labels",
]
labels_fixtures = seen_labels_fixtures + unseen_labels_fixtures


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("labels", lazy_fixture(labels_fixtures))
def test_inference_data_labels(labels: Labels) -> None:
    assert isinstance(labels, Labels)

    assert isinstance(labels.arxiv, pd.Series)
    assert labels.arxiv.name == "arxiv_labels"
    assert labels.arxiv.dtype == object
    assert labels.arxiv.index.name == "document_id"
    assert labels.arxiv.index.dtype == pd.Int64Dtype()
    assert all(isinstance(labels, list) for labels in labels.arxiv)
    assert all(isinstance(label, str) for labels in labels.arxiv for label in labels)
    assert all(len(labels) > 0 for labels in labels.arxiv)

    assert isinstance(labels.integer, pd.Series)
    assert labels.integer.name == "integer_labels"
    assert labels.integer.dtype == np.dtype("int64")
    assert labels.integer.index.name == "document_id"
    assert labels.integer.index.dtype == pd.Int64Dtype()


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("labels", lazy_fixture(seen_labels_fixtures))
def test_inference_data_seen_labels(labels: Labels) -> None:
    assert labels.integer.unique().tolist() == [0, 1]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("labels", lazy_fixture(unseen_labels_fixtures))
def test_inference_data_unseen_labels(labels: Labels) -> None:
    # for unseen papers no arxiv labels are available, thus there is no intersection
    # with the candidate paper arxiv labels
    assert labels.integer.unique().tolist() == [0]


# SECTION: Recommendations
recommendations_fixtures = [
    "inference_data_seen_recommendations",
    "inference_data_constructor_seen_recommendations",
    "inference_data_unseen_recommendations",
    "inference_data_constructor_unseen_recommendations",
]

recommendation_dataframe_fixtures_citation_to_language_candidates = [
    "inference_data_seen_recommendations_citation_to_language_candidates",
    "inference_data_unseen_recommendations_citation_to_language_candidates",
]
recommendation_dataframe_fixtures_citation_to_language = [
    "inference_data_seen_recommendations_citation_to_language",
    "inference_data_unseen_recommendations_citation_to_language",
]
recommendation_dataframe_fixtures_language_to_citation_candidates = [
    "inference_data_seen_recommendations_language_to_citation_candidates",
    "inference_data_unseen_recommendations_language_to_citation_candidates",
]
recommendation_dataframe_fixtures_language_to_citation = [
    "inference_data_seen_recommendations_language_to_citation",
    "inference_data_unseen_recommendations_language_to_citation",
]

recommendation_dataframe_fixtures_citation_features = (
    recommendation_dataframe_fixtures_citation_to_language_candidates
    + recommendation_dataframe_fixtures_language_to_citation
)

recommendation_dataframe_fixtures_language_features = (
    recommendation_dataframe_fixtures_language_to_citation_candidates
    + recommendation_dataframe_fixtures_citation_to_language
)

recommendation_dataframe_fixtures = (
    recommendation_dataframe_fixtures_citation_features
    + recommendation_dataframe_fixtures_language_features
)


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize("recommendations", lazy_fixture(recommendations_fixtures))
def test_inference_data_recommendations(recommendations: Recommendations) -> None:
    assert isinstance(recommendations, Recommendations)


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "recommendations_dataframe", lazy_fixture(recommendation_dataframe_fixtures)
)
def test_inference_data_recommendations_dataframes(recommendations_dataframe: pd.DataFrame) -> None:
    assert isinstance(recommendations_dataframe, pd.DataFrame)

    assert recommendations_dataframe.index.name == "document_id"
    assert recommendations_dataframe.index.dtype == pd.Int64Dtype()


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "recommendations_dataframe", lazy_fixture(recommendation_dataframe_fixtures_citation_features)
)
def test_inference_data_recommendations_dataframes_citation_candidates(
    recommendations_dataframe: pd.DataFrame,
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
        pd.StringDtype(),
        pd.StringDtype(),
        object,
        pd.StringDtype(),
        pd.Int64Dtype(),
        pd.Int64Dtype(),
        np.dtype("int64"),
        np.dtype("int64"),
    ]


@pytest.mark.slow
@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "recommendations_dataframe",
    lazy_fixture(recommendation_dataframe_fixtures_language_features),
)
def test_inference_data_recommendations_dataframes_language_candidates(
    recommendations_dataframe: pd.DataFrame,
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
        pd.StringDtype(),
        pd.StringDtype(),
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
                publication_date=pd.Series(),
                citationcount_document=pd.Series(),
                citationcount_author=pd.Series(),
                co_citation_analysis=pd.Series(),
                bibliographic_coupling=pd.Series(),
                cosine_similarity=pd.Series(),
                feature_weights=FeatureWeights(),
            ),
            Ranks(
                publication_date=pd.Series(),
                citationcount_document=pd.Series(),
                citationcount_author=pd.Series(),
                co_citation_analysis=pd.Series(),
                bibliographic_coupling=pd.Series(),
                cosine_similarity=pd.Series(),
            ),
            Labels(
                arxiv=pd.Series(),
                integer=pd.Series(),
            ),
            Recommendations(
                citation_to_language_candidates=pd.DataFrame(),
                citation_to_language=pd.DataFrame(),
                language_to_citation_candidates=pd.DataFrame(),
                language_to_citation=pd.DataFrame(),
            ),
        )
