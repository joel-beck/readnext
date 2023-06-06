import polars as pl
import pytest
import spacy
from spacy.language import Language
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.config import ModelVersions
from readnext.modeling.language_models import BERTTokenizer, LongformerTokenizer, SpacyTokenizer
from readnext.utils import DocumentsFrame, TokenIdsFrame, Tokens, TokensFrame


# SUBSECTION: SpaCy
# contained as dependency in pyproject.toml, can be used in CI
@pytest.fixture(scope="session")
def spacy_model() -> Language:
    return spacy.load(ModelVersions.spacy)


@pytest.fixture(scope="session")
def spacy_tokenizer(test_documents_frame: DocumentsFrame, spacy_model: Language) -> SpacyTokenizer:
    return SpacyTokenizer(test_documents_frame, spacy_model)


@pytest.fixture(scope="session")
def spacy_tokenized_abstracts() -> list[Tokens]:
    return [
        [
            "abstract",
            "example",
            "abstract",
            "character",
            "contain",
            "number",
            "special",
            "character",
            "like",
        ],
        ["abstract", "example", "abstract", "include", "upper", "case", "letter", "stopword"],
        [
            "abstract",
            "example",
            "abstract",
            "mix",
            "low",
            "case",
            "upper",
            "case",
            "letter",
            "punctuation",
            "bracket",
            "curly",
            "brace",
        ],
    ]


@pytest.fixture(scope="session")
def num_unique_corpus_tokens(spacy_tokenized_abstracts: list[Tokens]) -> int:
    # vocabulary has 18 unique tokens
    unique_corpus_tokens = {token for tokens in spacy_tokenized_abstracts for token in tokens}
    return len(unique_corpus_tokens)


@pytest.fixture(scope="session")
def spacy_tokens_frame(spacy_tokenized_abstracts: list[Tokens]) -> TokensFrame:
    return pl.from_records(
        list(enumerate(spacy_tokenized_abstracts)), schema=["d3_document_id", "tokens"]
    )


# SUBSECTION: BERT
@pytest.fixture(scope="session")
def bert_tokenizer_transformers() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )


@pytest.fixture(scope="session")
def bert_tokenizer(
    test_documents_frame: DocumentsFrame, bert_tokenizer_transformers: BertTokenizerFast
) -> BERTTokenizer:
    return BERTTokenizer(test_documents_frame, bert_tokenizer_transformers)


@pytest.fixture(scope="session")
def bert_token_ids_frame(bert_tokenizer: BERTTokenizer) -> TokenIdsFrame:
    return bert_tokenizer.tokenize()


@pytest.fixture(scope="session")
def bert_expected_tokenized_abstract() -> Tokens:
    return [
        "[CLS]",
        "abstract",
        "1",
        ":",
        "this",
        "is",
        "an",
        "example",
        "abstract",
        "with",
        "various",
        "characters",
        "!",
        "it",
        "contains",
        "numbers",
        "1",
        ",",
        "2",
        ",",
        "3",
        "and",
        "special",
        "characters",
        "like",
        "@",
        ",",
        "#",
        ",",
        "$",
        ".",
        "[SEP]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
    ]


# SUBSECTION: Longformer
@pytest.fixture(scope="session")
def longformer_tokenizer_transformers() -> LongformerTokenizerFast:
    return LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer, do_lower_case=True, clean_text=True
    )


@pytest.fixture(scope="session")
def longformer_tokenizer(
    test_documents_frame: DocumentsFrame, longformer_tokenizer_transformers: LongformerTokenizerFast
) -> LongformerTokenizer:
    return LongformerTokenizer(test_documents_frame, longformer_tokenizer_transformers)


@pytest.fixture(scope="session")
def longformer_token_ids_frame(longformer_tokenizer: LongformerTokenizer) -> TokenIdsFrame:
    return longformer_tokenizer.tokenize()


@pytest.fixture(scope="session")
def longformer_expected_tokenized_abstract() -> Tokens:
    return [
        "<s>",
        "Ċ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "ĠAbstract",
        "Ġ1",
        ":",
        "ĠThis",
        "Ġis",
        "Ġan",
        "Ġexample",
        "Ġabstract",
        "Ġwith",
        "Ġvarious",
        "Ġcharacters",
        "!",
        "ĠIt",
        "Ċ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġcontains",
        "Ġnumbers",
        "Ġ1",
        ",",
        "Ġ2",
        ",",
        "Ġ3",
        "Ġand",
        "Ġspecial",
        "Ġcharacters",
        "Ġlike",
        "Ġ@",
        ",",
        "Ġ#",
        ",",
        "Ġ$",
        ".",
        "Ċ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "Ġ",
        "</s>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
        "<pad>",
    ]
