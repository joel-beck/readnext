import pandas as pd

from readnext.modeling import (
    CitationModelDataConstructor,
    DocumentInfo,
    DocumentScore,
    LanguageModelDataConstructor,
)


# SECTION: Tests for CitationModelDataConstructor
def test_initialization(citation_model_data_constructor: CitationModelDataConstructor) -> None:
    assert isinstance(citation_model_data_constructor, CitationModelDataConstructor)

    assert isinstance(citation_model_data_constructor.d3_document_id, int)
    assert citation_model_data_constructor.d3_document_id == 546182

    assert isinstance(citation_model_data_constructor.documents_data, pd.DataFrame)
    assert citation_model_data_constructor.documents_data.shape[1] == 26

    assert isinstance(citation_model_data_constructor.co_citation_analysis_scores, pd.DataFrame)
    assert citation_model_data_constructor.co_citation_analysis_scores.shape[1] == 1

    assert isinstance(citation_model_data_constructor.bibliographic_coupling_scores, pd.DataFrame)
    assert citation_model_data_constructor.bibliographic_coupling_scores.shape[1] == 1

    assert isinstance(citation_model_data_constructor.query_document, DocumentInfo)


def test_collect_query_document(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> None:
    assert isinstance(citation_model_data_constructor.query_document.d3_document_id, int)
    assert citation_model_data_constructor.query_document.d3_document_id == 546182

    assert isinstance(citation_model_data_constructor.query_document.title, str)
    assert (
        citation_model_data_constructor.query_document.title
        == "Deep Residual Learning for Image Recognition"
    )

    assert isinstance(citation_model_data_constructor.query_document.author, str)
    assert citation_model_data_constructor.query_document.author == "Kaiming He"

    assert isinstance(citation_model_data_constructor.query_document.arxiv_labels, list)
    assert citation_model_data_constructor.query_document.arxiv_labels == ["cs.CV"]


def test_exclude_query_document(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> None:
    excluded_df = citation_model_data_constructor.exclude_query_document(
        citation_model_data_constructor.documents_data
    )

    assert isinstance(excluded_df, pd.DataFrame)
    assert citation_model_data_constructor.d3_document_id not in excluded_df.index


def test_filter_documents_data(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> None:
    filtered_df = citation_model_data_constructor.filter_documents_data()

    assert isinstance(filtered_df, pd.DataFrame)
    assert citation_model_data_constructor.d3_document_id not in filtered_df.index


def test_get_info_matrix(citation_model_data_constructor: CitationModelDataConstructor) -> None:
    info_matrix = citation_model_data_constructor.get_info_matrix()

    assert isinstance(info_matrix, pd.DataFrame)
    assert citation_model_data_constructor.d3_document_id not in info_matrix.index
    assert all(col in info_matrix.columns for col in citation_model_data_constructor.info_cols)


def test_extend_info_matrix(citation_model_data_constructor: CitationModelDataConstructor) -> None:
    # original query document id is not in citation scores data
    citation_model_data_constructor.d3_document_id = 206594692

    info_matrix = citation_model_data_constructor.get_info_matrix()
    extended_matrix = citation_model_data_constructor.extend_info_matrix(info_matrix)

    assert isinstance(extended_matrix, pd.DataFrame)
    assert all(col in extended_matrix.columns for col in citation_model_data_constructor.info_cols)


def test_shares_arxiv_label(citation_model_data_constructor: CitationModelDataConstructor) -> None:
    candidate_document_labels = ["cs.CV", "stat.ML"]
    result = citation_model_data_constructor.shares_arxiv_label(candidate_document_labels)

    assert isinstance(result, bool)
    assert result is True

    candidate_document_labels = ["cs.AI", "cs.LG"]
    result = citation_model_data_constructor.shares_arxiv_label(candidate_document_labels)

    assert isinstance(result, bool)
    assert result is False


def test_boolean_to_int(citation_model_data_constructor: CitationModelDataConstructor) -> None:
    result = citation_model_data_constructor.boolean_to_int(True)

    assert isinstance(result, int)
    assert result == 1


def test_get_integer_labels(citation_model_data_constructor: CitationModelDataConstructor) -> None:
    integer_labels = citation_model_data_constructor.get_integer_labels()

    assert isinstance(integer_labels, pd.Series)


def test_document_scores_to_frame(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> None:
    document_scores = [
        DocumentScore(
            document_info=DocumentInfo(
                d3_document_id=1, title="A", author="A.A", arxiv_labels=["cs.CV"]
            ),
            score=0.5,
        ),
        DocumentScore(
            document_info=DocumentInfo(
                d3_document_id=2, title="B", author="B.B", arxiv_labels=["stat.ML"]
            ),
            score=0.3,
        ),
    ]
    scores_df = citation_model_data_constructor.document_scores_to_frame(document_scores)

    assert isinstance(scores_df, pd.DataFrame)
    assert "score" in scores_df.columns
    assert scores_df.index.name == "document_id"


def test_get_citation_method_scores(
    citation_model_data_constructor_new_document_id: CitationModelDataConstructor,
) -> None:
    citation_method_data = (
        citation_model_data_constructor_new_document_id.co_citation_analysis_scores
    )
    scores_df = citation_model_data_constructor_new_document_id.get_citation_method_scores(
        citation_method_data
    )

    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape[1] == 1
    assert "score" in scores_df.columns
    assert scores_df.index.name == "document_id"


def test_get_co_citation_analysis_scores(
    citation_model_data_constructor_new_document_id: CitationModelDataConstructor,
) -> None:
    co_citation_analysis_scores = (
        citation_model_data_constructor_new_document_id.get_co_citation_analysis_scores()
    )

    assert isinstance(co_citation_analysis_scores, pd.DataFrame)
    assert co_citation_analysis_scores.shape[1] == 1
    assert "co_citation_analysis" in co_citation_analysis_scores.columns
    assert co_citation_analysis_scores.index.name == "document_id"


def test_get_bibliographic_coupling_scores(
    citation_model_data_constructor_new_document_id: CitationModelDataConstructor,
) -> None:
    bibliographic_coupling_scores = (
        citation_model_data_constructor_new_document_id.get_bibliographic_coupling_scores()
    )

    assert isinstance(bibliographic_coupling_scores, pd.DataFrame)
    assert bibliographic_coupling_scores.shape[1] == 1
    assert "bibliographic_coupling" in bibliographic_coupling_scores.columns
    assert bibliographic_coupling_scores.index.name == "document_id"


def test_extend_info_matrix_citation_model(
    citation_model_data_constructor_new_document_id: CitationModelDataConstructor,
) -> None:
    info_matrix = citation_model_data_constructor_new_document_id.get_info_matrix()
    extended_matrix = citation_model_data_constructor_new_document_id.extend_info_matrix(
        info_matrix
    )

    assert isinstance(extended_matrix, pd.DataFrame)
    assert (
        extended_matrix.shape[1]
        == len(citation_model_data_constructor_new_document_id.info_cols) + 2
    )
    assert "co_citation_analysis" in extended_matrix.columns
    assert "bibliographic_coupling" in extended_matrix.columns


def test_get_feature_matrix(
    citation_model_data_constructor_new_document_id: CitationModelDataConstructor,
) -> None:
    feature_matrix = citation_model_data_constructor_new_document_id.get_feature_matrix()

    assert isinstance(feature_matrix, pd.DataFrame)
    assert (
        feature_matrix.shape[1]
        == len(citation_model_data_constructor_new_document_id.feature_cols) + 2
    )
    assert "co_citation_analysis_rank" in feature_matrix.columns
    assert "bibliographic_coupling_rank" in feature_matrix.columns


# SECTION: Tests for LanguageModelDataConstructor
def test_get_cosine_similarity_scores(
    language_model_data_constructor_new_document_id: LanguageModelDataConstructor,
) -> None:
    scores_df = language_model_data_constructor_new_document_id.get_cosine_similarity_scores()

    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.shape[1] == 1
    assert "cosine_similarity" in scores_df.columns
    assert scores_df.index.name == "document_id"


def test_extend_info_matrix_language_model(
    language_model_data_constructor_new_document_id: LanguageModelDataConstructor,
) -> None:
    info_matrix = language_model_data_constructor_new_document_id.get_info_matrix()
    extended_matrix = language_model_data_constructor_new_document_id.extend_info_matrix(
        info_matrix
    )

    assert isinstance(extended_matrix, pd.DataFrame)
    assert (
        extended_matrix.shape[1]
        == len(language_model_data_constructor_new_document_id.info_cols) + 1
    )
    assert "cosine_similarity" in extended_matrix.columns


def test_get_cosine_similarity_ranks(
    language_model_data_constructor_new_document_id: LanguageModelDataConstructor,
) -> None:
    ranks_df = language_model_data_constructor_new_document_id.get_cosine_similarity_ranks()

    assert isinstance(ranks_df, pd.DataFrame)
    assert ranks_df.shape[1] == 1
    assert "cosine_similarity_rank" in ranks_df.columns
    assert ranks_df.index.name == "document_id"
