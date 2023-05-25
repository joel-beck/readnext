import pytest

from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
)


@pytest.fixture(scope="session")
def citation_model_data(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> CitationModelData:
    return CitationModelData.from_constructor(citation_model_data_constructor)


@pytest.fixture(scope="session")
def language_model_data(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> LanguageModelData:
    return LanguageModelData.from_constructor(language_model_data_constructor)
