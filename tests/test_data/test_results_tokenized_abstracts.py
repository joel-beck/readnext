import pytest

from readnext.config import ResultsPaths
from readnext.modeling.language_models import TokensIdMapping, TokensMapping
from readnext.utils import load_object_from_pickle, slice_mapping


@pytest.fixture(scope="module")
def spacy_tokenized_abstracts_mapping_most_cited() -> TokensMapping:
    return load_object_from_pickle(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="module")
def bert_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return load_object_from_pickle(
        ResultsPaths.language_models.bert_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="module")
def scibert_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return load_object_from_pickle(
        ResultsPaths.language_models.scibert_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="module")
def longformer_tokenized_abstracts_mapping_most_cited() -> TokensIdMapping:
    return load_object_from_pickle(
        ResultsPaths.language_models.longformer_tokenized_abstracts_mapping_most_cited_pkl
    )


def test_spacy_tokenized_abstracts_most_cited(
    spacy_tokenized_abstracts_mapping_most_cited: TokensMapping,
) -> None:
    assert isinstance(spacy_tokenized_abstracts_mapping_most_cited, dict)

    # check that keys are integers
    assert all(isinstance(key, int) for key in spacy_tokenized_abstracts_mapping_most_cited)

    # check that tokenized abstracts are lists of strings
    single_abstract = spacy_tokenized_abstracts_mapping_most_cited[206594692]
    assert isinstance(single_abstract, list)
    assert isinstance(single_abstract[0], str)

    # check that all tokenized abstracts are non-empty
    assert all(
        len(abstract) > 0 for abstract in spacy_tokenized_abstracts_mapping_most_cited.values()
    )


def test_bert_tokenized_abstracts_most_cited(
    bert_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
) -> None:
    assert isinstance(bert_tokenized_abstracts_mapping_most_cited, dict)

    # check that keys are integers
    assert all(isinstance(key, int) for key in bert_tokenized_abstracts_mapping_most_cited)

    # check that tokenized abstract ids are torch tensors
    single_abstract_ids = bert_tokenized_abstracts_mapping_most_cited[206594692]
    assert isinstance(single_abstract_ids, list)

    # check that tensors contain integer values
    assert isinstance(single_abstract_ids[0], int)

    # check that dimension of token ids tensor is 512
    assert len(single_abstract_ids) == 512

    # check the same for all abstracts
    assert all(
        len(abstract_ids) == 512
        for abstract_ids in bert_tokenized_abstracts_mapping_most_cited.values()
    )

    # check that first 254 token ids of first abstract are non-zero for bert tokenizer
    assert all(token_id != 0 for token_id in single_abstract_ids[:254])


def test_scibert_tokenized_abstracts_most_cited(
    scibert_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
) -> None:
    assert isinstance(scibert_tokenized_abstracts_mapping_most_cited, dict)

    # check that keys are integers
    assert all(isinstance(key, int) for key in scibert_tokenized_abstracts_mapping_most_cited)

    # check that tokenized abstract ids are lists of integers
    single_abstract_ids = scibert_tokenized_abstracts_mapping_most_cited[206594692]
    assert isinstance(single_abstract_ids, list)
    assert all(isinstance(token_id, int) for token_id in single_abstract_ids)

    # check that length of token ids is 512
    assert len(single_abstract_ids) == 512

    # check the same for all abstracts
    assert all(
        len(abstract_ids) == 512
        for abstract_ids in scibert_tokenized_abstracts_mapping_most_cited.values()
    )

    # check that first 252 token ids of first abstract are non-zero for scibert
    # tokenizer
    assert all(token_id != 0 for token_id in single_abstract_ids[:252])


def test_longformer_tokenized_abstracts_most_cited(
    longformer_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
) -> None:
    assert isinstance(longformer_tokenized_abstracts_mapping_most_cited, dict)

    # check that keys are integers
    assert all(isinstance(key, int) for key in longformer_tokenized_abstracts_mapping_most_cited)

    # check that tokenized abstract ids are lists of integers
    single_abstract_ids = longformer_tokenized_abstracts_mapping_most_cited[206594692]
    assert isinstance(single_abstract_ids, list)
    assert all(isinstance(token_id, int) for token_id in single_abstract_ids)

    # check that length of token ids is 1006
    assert len(single_abstract_ids) == 1006

    # check the same for all abstracts
    assert all(
        len(abstract_ids) == 1006
        for abstract_ids in longformer_tokenized_abstracts_mapping_most_cited.values()
    )
    # remaining tokens are not 0 but 1 for longformer! Check that from the 261st token
    # all remaining tokens are 1
    assert all(token_id == 1 for token_id in single_abstract_ids[261:])


def test_that_test_data_mimics_real_data(
    test_data_size: int,
    spacy_tokenized_abstracts_mapping_most_cited: TokensMapping,
    bert_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
    scibert_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
    longformer_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
    test_spacy_tokenized_abstracts_mapping_most_cited: TokensMapping,
    test_bert_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
    test_scibert_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
    test_longformer_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
) -> None:
    assert (
        slice_mapping(spacy_tokenized_abstracts_mapping_most_cited, test_data_size)
        == test_spacy_tokenized_abstracts_mapping_most_cited
    )

    assert (
        slice_mapping(bert_tokenized_abstracts_mapping_most_cited, test_data_size)
        == test_bert_tokenized_abstracts_mapping_most_cited
    )

    assert (
        slice_mapping(scibert_tokenized_abstracts_mapping_most_cited, test_data_size)
        == test_scibert_tokenized_abstracts_mapping_most_cited
    )

    assert (
        slice_mapping(longformer_tokenized_abstracts_mapping_most_cited, test_data_size)
        == test_longformer_tokenized_abstracts_mapping_most_cited
    )
