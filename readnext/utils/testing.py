import torch

from readnext.modeling.language_models import TokensTensorMapping


def assert_tokens_tensor_mapping_equal(
    mapping_1: TokensTensorMapping, mapping_2: TokensTensorMapping
) -> None:
    assert isinstance(mapping_1, dict)
    assert isinstance(mapping_2, dict)

    assert all(isinstance(key, int) for key in mapping_1)
    assert all(isinstance(key, int) for key in mapping_2)

    # test that keys are equal
    assert set(mapping_1.keys()) == set(mapping_2.keys())

    assert all(isinstance(value, torch.Tensor) for value in mapping_1.values())
    assert all(isinstance(value, torch.Tensor) for value in mapping_2.values())

    assert all(isinstance(value[0].item(), int) for value in mapping_1.values())
    assert all(isinstance(value[0].item(), int) for value in mapping_2.values())

    # test that values are equal
    assert all(
        torch.equal(value_1, value_2)
        for value_1, value_2 in zip(mapping_1.values(), mapping_2.values())
    )
