import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.modeling.language_models import TokensIdMapping, TokensMapping

bert_based_tokenized_abstracts_mappings = [
    "bert_tokenized_abstracts_mapping_most_cited",
    "scibert_tokenized_abstracts_mapping_most_cited",
]

torch_tokenized_abstracts_mappings = [
    *bert_based_tokenized_abstracts_mappings,
    "longformer_tokenized_abstracts_mapping_most_cited",
]

all_tokenized_abstracts_mappings = [
    *torch_tokenized_abstracts_mappings,
    "spacy_tokenized_abstracts_mapping_most_cited",
]


@pytest.mark.parametrize(
    "tokenized_abstracts_mapping", lazy_fixture(all_tokenized_abstracts_mappings)
)
def test_key_values_tokenized_abstracts_mappings_most_cited(
    tokenized_abstracts_mapping: TokensMapping,
) -> None:
    assert isinstance(tokenized_abstracts_mapping, dict)

    # check that keys are integers and values are lists
    assert all(isinstance(key, int) for key in tokenized_abstracts_mapping)
    assert all(isinstance(value, list) for value in tokenized_abstracts_mapping.values())


def test_spacy_tokenized_abstract_mappings_most_cited(
    spacy_tokenized_abstracts_mapping_most_cited: TokensMapping,
) -> None:
    # check that tokenized abstracts are lists of strings
    single_abstract = spacy_tokenized_abstracts_mapping_most_cited[206594692]
    assert all(isinstance(token, str) for token in single_abstract)

    # check that all tokenized abstracts are non-empty
    assert all(
        len(abstract) > 0 for abstract in spacy_tokenized_abstracts_mapping_most_cited.values()
    )


@pytest.mark.parametrize(
    "tokenized_abstract_mapping", lazy_fixture(torch_tokenized_abstracts_mappings)
)
def test_torch_tokenized_abstract_mappings_token_ids(
    tokenized_abstract_mapping: TokensIdMapping,
) -> None:
    # check that tokenized abstract are lists of integers
    single_abstract_token_ids = tokenized_abstract_mapping[206594692]
    assert all(isinstance(token_id, int) for token_id in single_abstract_token_ids)


@pytest.mark.parametrize(
    "tokenized_abstract_mapping", lazy_fixture(bert_based_tokenized_abstracts_mappings)
)
def test_bert_based_tokenized_abstracts_length(tokenized_abstract_mapping: TokensIdMapping) -> None:
    # check that tokenized abstracts have length 512
    assert all(len(abstract_ids) == 512 for abstract_ids in tokenized_abstract_mapping.values())


def test_bert_tokenized_abstracts_most_cited(
    bert_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
) -> None:
    single_abstract_token_ids = bert_tokenized_abstracts_mapping_most_cited[206594692]
    # check that first 254 token ids of first abstract are non-zero for bert tokenizer
    assert all(token_id != 0 for token_id in single_abstract_token_ids[:254])


def test_scibert_tokenized_abstracts_most_cited(
    scibert_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
) -> None:
    single_abstract_token_ids = scibert_tokenized_abstracts_mapping_most_cited[206594692]
    # check that first 252 token ids of first abstract are non-zero for scibert
    # tokenizer
    assert all(token_id != 0 for token_id in single_abstract_token_ids[:252])


def test_longformer_tokenized_abstracts_most_cited(
    longformer_tokenized_abstracts_mapping_most_cited: TokensIdMapping,
) -> None:
    # check that tokenized abstracts have length 1006
    assert all(
        len(abstract_ids) == 1006
        for abstract_ids in longformer_tokenized_abstracts_mapping_most_cited.values()
    )

    single_abstract_token_ids = longformer_tokenized_abstracts_mapping_most_cited[206594692]
    # remaining tokens are not 0 but 1 for longformer! Check that from the 261st token
    # all remaining tokens are 1
    assert all(token_id == 1 for token_id in single_abstract_token_ids[261:])
