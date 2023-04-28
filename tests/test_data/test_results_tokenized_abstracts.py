import pytest
import torch

from readnext.config import ResultsPaths
from readnext.utils import load_object_from_pickle


@pytest.fixture(scope="module")
def spacy_tokenized_abstracts_mapping_most_cited() -> dict[int, list[str]]:
    return load_object_from_pickle(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl
    )


@pytest.fixture(scope="module")
def bert_tokenized_abstracts_mapping_most_cited() -> dict[int, torch.Tensor]:
    return torch.load(ResultsPaths.language_models.bert_tokenized_abstracts_mapping_most_cited_pt)


@pytest.fixture(scope="module")
def scibert_tokenized_abstracts_mapping_most_cited() -> dict[int, torch.Tensor]:
    return torch.load(
        ResultsPaths.language_models.scibert_tokenized_abstracts_mapping_most_cited_pt
    )


def test_spacy_tokenized_abstracts_most_cited(
    spacy_tokenized_abstracts_mapping_most_cited: dict[int, list[str]]
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
    bert_tokenized_abstracts_mapping_most_cited: dict[int, torch.Tensor]
) -> None:
    assert isinstance(bert_tokenized_abstracts_mapping_most_cited, dict)

    # check that keys are integers
    assert all(isinstance(key, int) for key in bert_tokenized_abstracts_mapping_most_cited)

    # check that tokenized abstract ids are torch tensors
    single_abstract_ids = bert_tokenized_abstracts_mapping_most_cited[206594692]
    assert isinstance(single_abstract_ids, torch.Tensor)

    # check that tensors contain integer values
    assert isinstance(single_abstract_ids[0].item(), int)

    # check that dimension of token ids tensor is 512
    assert single_abstract_ids.size()[0] == 512

    # check the same for all abstracts
    assert all(
        abstract_ids.size()[0] == 512
        for abstract_ids in bert_tokenized_abstracts_mapping_most_cited.values()
    )

    # check that first 254 token ids of first abstract are non-zero for bert tokenizer
    assert single_abstract_ids.nonzero().size()[0] == 254


def test_scibert_tokenized_abstracts_most_cited(
    scibert_tokenized_abstracts_mapping_most_cited: dict[int, torch.Tensor]
) -> None:
    assert isinstance(scibert_tokenized_abstracts_mapping_most_cited, dict)

    # check that keys are integers
    assert all(isinstance(key, int) for key in scibert_tokenized_abstracts_mapping_most_cited)

    # check that tokenized abstract ids are torch tensors
    single_abstract_ids = scibert_tokenized_abstracts_mapping_most_cited[206594692]
    assert isinstance(single_abstract_ids, torch.Tensor)

    # check that tensors contain integer values
    assert isinstance(single_abstract_ids[0].item(), int)

    # check that dimension of token ids tensor is 512
    assert single_abstract_ids.size()[0] == 512

    # check the same for all abstracts
    assert all(
        abstract_ids.size()[0] == 512
        for abstract_ids in scibert_tokenized_abstracts_mapping_most_cited.values()
    )

    # check that first 252 token ids of first abstract are non-zero for scibert
    # tokenizer
    assert single_abstract_ids.nonzero().size()[0] == 252
