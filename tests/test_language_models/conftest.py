import pytest
import spacy
from spacy.language import Language
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.config import ModelVersions
from readnext.modeling import (
    DocumentInfo,
    DocumentsInfo,
)
from readnext.modeling.language_models import (
    BERTEmbedder,
    BERTTokenizer,
    FastTextEmbedder,
    LongformerEmbedder,
    LongformerTokenizer,
    SpacyTokenizer,
    TFIDFEmbedder,
    Word2VecEmbedder,
    tfidf,
)
from readnext.utils import (
    BertModelProtocol,
    FastTextModelProtocol,
    LongformerModelProtocol,
    Tokens,
    TokensMapping,
    Word2VecModelProtocol,
)
from readnext.utils.aliases import TokensIdMapping


# SECTION: Tokens
@pytest.fixture(scope="session")
def document_tokens() -> Tokens:
    return ["a", "b", "c", "a", "b", "c", "d", "d", "d"]


@pytest.fixture(scope="session")
def document_corpus() -> list[Tokens]:
    return [
        ["a", "b", "c", "d", "d", "d"],
        ["a", "b", "b", "c", "c", "c", "d"],
        ["a", "a", "a", "b", "c", "d"],
    ]


@pytest.fixture(scope="session")
def documents_info() -> DocumentsInfo:
    return DocumentsInfo(
        [
            DocumentInfo(
                d3_document_id=1,
                title="Title 1",
                author="Author 1",
                abstract="""
                Abstract 1: This is an example abstract with various characters! It
                contains numbers 1, 2, 3 and special characters like @, #, $.
                """,
            ),
            DocumentInfo(
                d3_document_id=2,
                title="Title 2",
                author="Author 2",
                abstract="""
                Abstract 2: Another example abstract, including upper-case letters and a
                few stopwords such as 'the', 'and', 'in'.
                """,
            ),
            DocumentInfo(
                d3_document_id=3,
                title="Title 3",
                author="Author 3",
                abstract="""
                Abstract 3: A third example abstract with a mix of lower-case and
                UPPER-CASE letters, as well as some punctuation: (brackets) and {curly
                braces}.
                """,
            ),
        ]
    )


# SECTION: Tokenizer
# SUBSECTION: SpaCy
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
def bert_tokens_id_mapping(bert_tokenizer: BERTTokenizer) -> TokensIdMapping:
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
def longformer_tokens_id_mapping(longformer_tokenizer: LongformerTokenizer) -> TokensIdMapping:
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


# SECTION: Embedders
@pytest.fixture(scope="session")
def tfidf_embedder(spacy_tokens_mapping: TokensMapping) -> TFIDFEmbedder:
    return TFIDFEmbedder(keyword_algorithm=tfidf, tokens_mapping=spacy_tokens_mapping)


@pytest.fixture(scope="session")
def word2vec_embedder(
    spacy_tokens_mapping: TokensMapping, word2vec_model: Word2VecModelProtocol
) -> Word2VecEmbedder:
    return Word2VecEmbedder(spacy_tokens_mapping, word2vec_model)


@pytest.fixture(scope="session")
def fasttext_embedder(
    spacy_tokens_mapping: TokensMapping, fasttext_model: FastTextModelProtocol
) -> FastTextEmbedder:
    return FastTextEmbedder(spacy_tokens_mapping, fasttext_model)


# TODO: Unify Order of tokens mapping and model across all embedders
@pytest.fixture(scope="session")
def bert_embedder(
    bert_tokens_id_mapping: TokensIdMapping, bert_model: BertModelProtocol
) -> BERTEmbedder:
    return BERTEmbedder(bert_model, bert_tokens_id_mapping)


@pytest.fixture(scope="session")
def longformer_embedder(
    longformer_tokens_id_mapping: TokensIdMapping, longformer_model: LongformerModelProtocol
) -> LongformerEmbedder:
    return LongformerEmbedder(longformer_model, longformer_tokens_id_mapping)
