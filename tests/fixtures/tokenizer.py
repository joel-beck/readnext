import pytest
import spacy
from spacy.language import Language
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.config import ModelVersions
from readnext.modeling.language_models import BERTTokenizer, LongformerTokenizer, SpacyTokenizer


# contained as dependency in pyproject.toml, can be used in CI
@pytest.fixture(scope="session")
def spacy_model() -> Language:
    return spacy.load(ModelVersions.spacy)


@pytest.fixture(scope="session")
def spacy_tokenizer(spacy_model: Language) -> SpacyTokenizer:
    return SpacyTokenizer(spacy_model)


@pytest.fixture(scope="session")
def bert_tokenizer_transformers() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )


@pytest.fixture(scope="session")
def bert_tokenizer(bert_tokenizer_transformers: BertTokenizerFast) -> BERTTokenizer:
    return BERTTokenizer(bert_tokenizer_transformers)


@pytest.fixture(scope="session")
def longformer_tokenizer_transformers() -> LongformerTokenizerFast:
    return LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer, do_lower_case=True, clean_text=True
    )


@pytest.fixture(scope="session")
def longformer_tokenizer(
    longformer_tokenizer_transformers: LongformerTokenizerFast,
) -> LongformerTokenizer:
    return LongformerTokenizer(longformer_tokenizer_transformers)
