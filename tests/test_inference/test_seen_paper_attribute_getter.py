from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from readnext.evaluation.scoring import FeatureWeights
from readnext.inference import DocumentIdentifier, InferenceDataInputConverter
from readnext.inference.attribute_getter import SeenPaperAttributeGetter
from readnext.modeling.language_models.model_choice import LanguageModelChoice
from readnext.utils import load_df_from_pickle


def test_documents_authors_labels_citations_most_cited() -> pd.DataFrame:
    return load_df_from_pickle(
        Path("./../data/test_documents_authors_labels_citations_most_cited.pkl")
    )


# NOTE: Make sure the test document is also seen in the TEST documents data, i.e. within
# the e.g. top 100 most cited papers
semanticscholar_url = (
    "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
)

seen_paper_attribute_getter = SeenPaperAttributeGetter(
    semanticscholar_url=semanticscholar_url,
    semanticscholar_id=None,
    arxiv_id=None,
    arxiv_url=None,
    language_model_choice=LanguageModelChoice.tfidf,
    feature_weights=FeatureWeights(),
    documents_data=test_documents_authors_labels_citations_most_cited(),
)


def input_converter_sets_identifier_correctly(
    seen_paper_attribute_getter: SeenPaperAttributeGetter,
) -> None:
    assert isinstance(seen_paper_attribute_getter.identifier, DocumentIdentifier)
    assert seen_paper_attribute_getter.identifier.d3_document_id is not None
    assert seen_paper_attribute_getter.identifier.semanticscholar_id is not None
    assert seen_paper_attribute_getter.identifier.semanticscholar_url is not None
    assert seen_paper_attribute_getter.identifier.arxiv_id is not None
    assert seen_paper_attribute_getter.identifier.arxiv_url is not None


def test_input_converter_works_for_semanticscholar_url(
    seen_paper_attribute_getter_semanticscholar_url: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_semanticscholar_url.semanticscholar_url is not None
    assert isinstance(seen_paper_attribute_getter_semanticscholar_url.semanticscholar_url, str)
    assert seen_paper_attribute_getter_semanticscholar_url.semanticscholar_id is None
    assert seen_paper_attribute_getter_semanticscholar_url.arxiv_id is None
    assert seen_paper_attribute_getter_semanticscholar_url.arxiv_url is None

    input_converter_sets_identifier_correctly(seen_paper_attribute_getter_semanticscholar_url)


def test_input_converter_works_for_semanticscholar_id(
    seen_paper_attribute_getter_semanticscholar_id: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_semanticscholar_id.semanticscholar_url is not None
    assert isinstance(seen_paper_attribute_getter_semanticscholar_id.semanticscholar_url, str)
    assert seen_paper_attribute_getter_semanticscholar_id.semanticscholar_id is None
    assert seen_paper_attribute_getter_semanticscholar_id.arxiv_id is None
    assert seen_paper_attribute_getter_semanticscholar_id.arxiv_url is None

    input_converter_sets_identifier_correctly(seen_paper_attribute_getter_semanticscholar_id)


def test_input_converter_works_for_arxiv_id(
    seen_paper_attribute_getter_arxiv_id: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_arxiv_id.semanticscholar_url is not None
    assert isinstance(seen_paper_attribute_getter_arxiv_id.semanticscholar_url, str)
    assert seen_paper_attribute_getter_arxiv_id.semanticscholar_id is None
    assert seen_paper_attribute_getter_arxiv_id.arxiv_id is None
    assert seen_paper_attribute_getter_arxiv_id.arxiv_url is None

    input_converter_sets_identifier_correctly(seen_paper_attribute_getter_arxiv_id)


def test_input_converter_works_for_arxiv_url(
    seen_paper_attribute_getter_arxiv_url: SeenPaperAttributeGetter,
) -> None:
    assert seen_paper_attribute_getter_arxiv_url.semanticscholar_url is not None
    assert isinstance(seen_paper_attribute_getter_arxiv_url.semanticscholar_url, str)
    assert seen_paper_attribute_getter_arxiv_url.semanticscholar_id is None
    assert seen_paper_attribute_getter_arxiv_url.arxiv_id is None
    assert seen_paper_attribute_getter_arxiv_url.arxiv_url is None

    input_converter_sets_identifier_correctly(seen_paper_attribute_getter_arxiv_url)


def test_attributes_are_created_correctly() -> None:
    assert isinstance(seen_paper_attribute_getter.input_converter, InferenceDataInputConverter)
    assert isinstance(seen_paper_attribute_getter.documents_data, pd.DataFrame)
    assert isinstance(seen_paper_attribute_getter.input_converter.documents_data, pd.DataFrame)
    # same data attribute is passed to the input converter
    assert_frame_equal(
        seen_paper_attribute_getter.documents_data,
        seen_paper_attribute_getter.input_converter.documents_data,
    )
