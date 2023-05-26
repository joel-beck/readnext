from spacy.language import Language
from spacy.tokens.doc import Doc

from readnext.modeling import DocumentInfo, DocumentsInfo
from readnext.modeling.language_models import SpacyTokenizer


def test_to_spacy_doc(spacy_tokenizer: SpacyTokenizer, documents_info: DocumentsInfo) -> None:
    assert all(
        isinstance(spacy_tokenizer.to_spacy_doc(abstract), Doc)
        for abstract in documents_info.abstracts
    )


def test_clean_spacy_doc(
    spacy_tokenizer: SpacyTokenizer,
    documents_info: DocumentsInfo,
    spacy_tokenized_abstracts: list[list[str]],
) -> None:
    docs = [spacy_tokenizer.to_spacy_doc(abstract) for abstract in documents_info.abstracts]

    docs_clean = [spacy_tokenizer.clean_spacy_doc(doc) for doc in docs]

    assert all(isinstance(doc, list) for doc in docs_clean)
    assert all(isinstance(token, str) for doc in docs_clean for token in doc)

    assert docs_clean == spacy_tokenized_abstracts


def test_tokenize(
    spacy_tokenizer: SpacyTokenizer, spacy_tokenized_abstracts: list[list[str]]
) -> None:
    tokens_mapping = spacy_tokenizer.tokenize()
    assert isinstance(tokens_mapping, dict)

    assert all(isinstance(key, int) for key in tokens_mapping)
    assert all(isinstance(value, list) for value in tokens_mapping.values())
    assert all(isinstance(token, str) for value in tokens_mapping.values() for token in value)

    assert list(tokens_mapping.keys()) == [1, 2, 3]
    assert list(tokens_mapping.values()) == spacy_tokenized_abstracts


def test_tokenize_empty_abstract(spacy_model: Language) -> None:
    documents_info = DocumentsInfo(
        [
            DocumentInfo(d3_document_id=1, title="Empty paper", abstract=""),
        ]
    )

    spacy_tokenizer = SpacyTokenizer(documents_info, spacy_model)
    tokenized_abstracts_mapping = spacy_tokenizer.tokenize()
    assert tokenized_abstracts_mapping[1] == []


def test_tokenize_stopwords(spacy_tokenizer: SpacyTokenizer) -> None:
    stopwords = set(spacy_tokenizer.spacy_model.Defaults.stop_words)
    tokenized_abstracts_mapping = spacy_tokenizer.tokenize()

    for tokens in tokenized_abstracts_mapping.values():
        assert all(token not in stopwords for token in tokens)


def test_tokenize_punctuation(spacy_tokenizer: SpacyTokenizer) -> None:
    tokenized_abstracts_mapping = spacy_tokenizer.tokenize()

    for tokens in tokenized_abstracts_mapping.values():
        assert all(token.isalnum() for token in tokens)


def test_tokenize_lowercase(spacy_tokenizer: SpacyTokenizer) -> None:
    tokenized_abstracts_mapping = spacy_tokenizer.tokenize()

    for tokens in tokenized_abstracts_mapping.values():
        assert all(token.islower() for token in tokens)
