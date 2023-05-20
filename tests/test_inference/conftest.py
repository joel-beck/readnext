import pandas as pd
import pytest

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference.attribute_getter import SeenPaperAttributeGetter
from readnext.modeling.language_models.model_choice import LanguageModelChoice


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_semanticscholar_id(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
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
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_semanticscholar_url(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
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
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_arxiv_id(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> SeenPaperAttributeGetter:
    arxiv_id = "1706.03762"

    return SeenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=arxiv_id,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_from_arxiv_url(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> SeenPaperAttributeGetter:
    arxiv_url = "https://arxiv.org/abs/1706.03762"

    return SeenPaperAttributeGetter(
        semanticscholar_id=None,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=arxiv_url,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )
