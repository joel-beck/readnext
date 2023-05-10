import pandas as pd

from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
)


def test_citation_model_data_from_constructor(
    citation_model_data_constructor_new_document_id: CitationModelDataConstructor,
) -> None:
    citation_model_data = CitationModelData.from_constructor(
        citation_model_data_constructor_new_document_id
    )

    assert isinstance(citation_model_data, CitationModelData)
    assert isinstance(citation_model_data.info_matrix, pd.DataFrame)
    assert isinstance(citation_model_data.integer_labels, pd.Series)
    assert isinstance(citation_model_data.feature_matrix, pd.DataFrame)


def test_citation_model_data_getitem(
    test_data_size: int,
    citation_model_data_constructor_new_document_id: CitationModelDataConstructor,
) -> None:
    citation_model_data = CitationModelData.from_constructor(
        citation_model_data_constructor_new_document_id
    )
    index_info_matrix = citation_model_data.info_matrix.index
    index_feature_matrix = citation_model_data.feature_matrix.index
    shared_indices = index_info_matrix.intersection(index_feature_matrix)

    sliced_citation_model_data = citation_model_data[shared_indices]

    assert isinstance(sliced_citation_model_data, CitationModelData)
    # index of info matrix and feature matrix is identical
    assert len(sliced_citation_model_data.info_matrix) == test_data_size
    assert len(sliced_citation_model_data.integer_labels) == test_data_size
    assert len(sliced_citation_model_data.feature_matrix) == test_data_size


def test_language_model_data_from_constructor(
    language_model_data_constructor_new_document_id: LanguageModelDataConstructor,
) -> None:
    language_model_data = LanguageModelData.from_constructor(
        language_model_data_constructor_new_document_id
    )

    assert isinstance(language_model_data, LanguageModelData)
    assert isinstance(language_model_data.info_matrix, pd.DataFrame)
    assert isinstance(language_model_data.integer_labels, pd.Series)
    assert isinstance(language_model_data.cosine_similarity_ranks, pd.DataFrame)


def test_language_model_data_getitem(
    language_model_data_constructor_new_document_id: LanguageModelDataConstructor,
) -> None:
    language_model_data = LanguageModelData.from_constructor(
        language_model_data_constructor_new_document_id
    )

    index_info_matrix = language_model_data.info_matrix.index
    index_cosine_similarity_ranks = language_model_data.cosine_similarity_ranks.index
    shared_indices = index_info_matrix.intersection(index_cosine_similarity_ranks)

    sliced_language_model_data = language_model_data[shared_indices]

    assert isinstance(sliced_language_model_data, LanguageModelData)
    # index of info matrix and cosine similarity ranks is not identical! (here special
    # case of no overlap)
    assert len(sliced_language_model_data.info_matrix) == 0
    assert len(sliced_language_model_data.integer_labels) == 0
    assert len(sliced_language_model_data.cosine_similarity_ranks) == 0
