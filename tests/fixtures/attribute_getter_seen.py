import polars as pl
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference.attribute_getter import SeenPaperAttributeGetter
from readnext.modeling import (
    CitationModelData,
    DocumentInfo,
    LanguageModelData,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils import ScoresFrame


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_semanticscholar_id(
    test_documents_data: pl.DataFrame,
) -> SeenPaperAttributeGetter:
    # NOTE: Make sure the test document is also seen in the TEST documents data, i.e. within
    # the e.g. top 100 most cited papers
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    return SeenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_semanticscholar_url(
    test_documents_data: pl.DataFrame,
) -> SeenPaperAttributeGetter:
    semanticscholar_url = (
        "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    )

    return SeenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=semanticscholar_url,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_arxiv_id(
    test_documents_data: pl.DataFrame,
) -> SeenPaperAttributeGetter:
    arxiv_id = "1706.03762"

    return SeenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=arxiv_id,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_arxiv_url(
    test_documents_data: pl.DataFrame,
) -> SeenPaperAttributeGetter:
    arxiv_url = "https://arxiv.org/abs/1706.03762"

    return SeenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_co_citation_analysis(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_co_citation_analysis_scores()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_bibliographic_coupling(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_bibliographic_coupling_scores()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_tfidf(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_bm25(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.bm25,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_word2vec(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.word2vec,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_glove(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.glove,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_fasttext(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.fasttext,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_bert(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.bert,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_scibert(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.scibert,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_longformer(
    test_documents_data: pl.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.longformer,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data(
    test_documents_data: pl.DataFrame,
) -> CitationModelData:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_citation_model_data()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data_query_document(
    seen_paper_attribute_getter_citation_model_data: CitationModelData,
) -> DocumentInfo:
    return seen_paper_attribute_getter_citation_model_data.query_document


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data_integer_labels(
    seen_paper_attribute_getter_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_citation_model_data.integer_labels


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data_info_matrix(
    seen_paper_attribute_getter_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_citation_model_data.info_matrix


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data_feature_matrix(
    seen_paper_attribute_getter_citation_model_data: CitationModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_citation_model_data.feature_matrix


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data(
    test_documents_data: pl.DataFrame,
) -> LanguageModelData:
    semanticscholar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_data,
    )

    return seen_paper_attribute_getter.get_language_model_data()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data_query_document(
    seen_paper_attribute_getter_language_model_data: LanguageModelData,
) -> DocumentInfo:
    return seen_paper_attribute_getter_language_model_data.query_document


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data_integer_labels(
    seen_paper_attribute_getter_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_language_model_data.integer_labels


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data_info_matrix(
    seen_paper_attribute_getter_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_language_model_data.info_matrix


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data_cosine_similarity_ranks(
    seen_paper_attribute_getter_language_model_data: LanguageModelData,
) -> pl.DataFrame:
    return seen_paper_attribute_getter_language_model_data.cosine_similarity_ranks
