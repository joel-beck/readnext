import pytest
import spacy
from spacy.language import Language
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.config import ModelVersions
from readnext.modeling import DocumentsInfo
from readnext.modeling.language_models import BERTTokenizer, LongformerTokenizer, SpacyTokenizer
from readnext.utils import Tokens, TokensMapping
from readnext.utils.aliases import TokenIdsMapping


# SUBSECTION: SpaCy
# contained as dependency in pyproject.toml, can be used in CI
@pytest.fixture(scope="session")
def spacy_model() -> Language:
    return spacy.load(ModelVersions.spacy)


@pytest.fixture(scope="session")
def spacy_tokenizer(documents_info: DocumentsInfo, spacy_model: Language) -> SpacyTokenizer:
    return SpacyTokenizer(documents_info, spacy_model)


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
def spacy_tokens_mapping(spacy_tokenized_abstracts: list[Tokens]) -> TokensMapping:
    return dict(enumerate(spacy_tokenized_abstracts))


# SUBSECTION: BERT
@pytest.fixture(scope="session")
def bert_tokenizer_transformers() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )


@pytest.fixture(scope="session")
def bert_tokenizer(
    documents_info: DocumentsInfo, bert_tokenizer_transformers: BertTokenizerFast
) -> BERTTokenizer:
    return BERTTokenizer(documents_info, bert_tokenizer_transformers)


@pytest.fixture(scope="session")
def bert_tokens_id_mapping(bert_tokenizer: BERTTokenizer) -> TokenIdsMapping:
    return bert_tokenizer.tokenize()


@pytest.fixture(scope="session")
def bert_exptected_tokenized_abstract() -> Tokens:
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
    documents_info: DocumentsInfo, longformer_tokenizer_transformers: LongformerTokenizerFast
) -> LongformerTokenizer:
    return LongformerTokenizer(documents_info, longformer_tokenizer_transformers)


@pytest.fixture(scope="session")
def longformer_tokens_id_mapping(longformer_tokenizer: LongformerTokenizer) -> TokenIdsMapping:
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
