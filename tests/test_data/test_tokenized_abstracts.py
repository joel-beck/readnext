import pytest
from pytest_lazyfixture import lazy_fixture

from readnext.utils import TokenIdsFrame, TokensFrame

bert_based_tokenized_abstracts_mappings = [
    "bert_tokenized_abstracts_mapping",
    "scibert_tokenized_abstracts_mapping",
]

torch_tokenized_abstracts_mappings = [
    *bert_based_tokenized_abstracts_mappings,
    "longformer_tokenized_abstracts_mapping",
]

all_tokenized_abstracts_mappings = [
    *torch_tokenized_abstracts_mappings,
    "spacy_tokenized_abstracts_mapping",
]


@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "tokenized_abstracts_mapping", lazy_fixture(all_tokenized_abstracts_mappings)
)
def test_key_values_tokenized_abstracts_mappings(
    tokenized_abstracts_mapping: TokensFrame,
) -> None:
    assert isinstance(tokenized_abstracts_mapping, dict)

    # check that keys are integers and values are lists
    assert all(isinstance(key, int) for key in tokenized_abstracts_mapping)
    assert all(isinstance(value, list) for value in tokenized_abstracts_mapping.values())


@pytest.mark.skip_ci
def test_spacy_tokenized_abstract_mappings(
    spacy_tokenized_abstracts_mapping: TokensFrame,
) -> None:
    # check that tokenized abstracts are lists of strings
    single_abstract = spacy_tokenized_abstracts_mapping[13756489]
    assert all(isinstance(token, str) for token in single_abstract)

    # check that all tokenized abstracts are non-empty
    assert all(len(abstract) > 0 for abstract in spacy_tokenized_abstracts_mapping.values())


@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "tokenized_abstract_mapping", lazy_fixture(torch_tokenized_abstracts_mappings)
)
def test_torch_tokenized_abstract_mappings_token_ids(
    tokenized_abstract_mapping: TokenIdsFrame,
) -> None:
    # check that tokenized abstract are lists of integers
    single_abstract_token_ids = tokenized_abstract_mapping[13756489]
    assert all(isinstance(token_id, int) for token_id in single_abstract_token_ids)


@pytest.mark.skip_ci
@pytest.mark.parametrize(
    "tokenized_abstract_mapping", lazy_fixture(bert_based_tokenized_abstracts_mappings)
)
def test_bert_based_tokenized_abstracts_length(tokenized_abstract_mapping: TokenIdsFrame) -> None:
    # check that tokenized abstracts have length 512
    assert all(len(abstract_ids) == 512 for abstract_ids in tokenized_abstract_mapping.values())


@pytest.mark.skip_ci
def test_bert_tokenized_abstracts(
    bert_tokenized_abstracts_mapping: TokenIdsFrame,
) -> None:
    single_abstract_token_ids = bert_tokenized_abstracts_mapping[13756489]
    # only padding tokens from index 242 onwards
    assert sum(token_id != 0 for token_id in single_abstract_token_ids[:242])
    assert all(token_id == 0 for token_id in single_abstract_token_ids[242:])


@pytest.mark.skip_ci
def test_scibert_tokenized_abstracts(
    scibert_tokenized_abstracts_mapping: TokenIdsFrame,
) -> None:
    single_abstract_token_ids = scibert_tokenized_abstracts_mapping[13756489]
    # only padding tokens from index 217 onwards
    assert all(token_id != 0 for token_id in single_abstract_token_ids[:217])
    assert all(token_id == 0 for token_id in single_abstract_token_ids[217:])


@pytest.mark.skip_ci
def test_longformer_tokenized_abstracts(
    longformer_tokenized_abstracts_mapping: TokenIdsFrame,
) -> None:
    # check that tokenized abstracts have length 1006
    assert all(
        len(abstract_ids) == 1006
        for abstract_ids in longformer_tokenized_abstracts_mapping.values()
    )

    single_abstract_token_ids = longformer_tokenized_abstracts_mapping[13756489]
    # only padding tokens from index 217 onwards
    assert all(token_id == 1 for token_id in single_abstract_token_ids[261:])
