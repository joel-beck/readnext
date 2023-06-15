# TODO: Write new tests from scratch


def test_tokenize() -> None:
    # test that tokenization of a dummy abstract leads to expected tokens without stopwords etc.
    ...


def test_tokenize_empty_abstract() -> None:
    ...


def test_tokenize_stopwords() -> None:
    ...
    # test that all stopwords are removed
    # stopwords = set(spacy_tokenizer.spacy_model.Defaults.stop_words)


def test_tokenize_punctuation() -> None:
    ...
    # test that all punctuation is removed
    # assert all(token.isalnum() for token in tokens)


def test_tokenize_lowercase() -> None:
    ...
    # test that all tokens are lowercased
    # assert all(token.islower() for token in tokens)
