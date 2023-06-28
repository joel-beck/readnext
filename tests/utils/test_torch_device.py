import pytest
import torch

from readnext.utils.torch_device import get_torch_device


def test_get_torch_device_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "has_cuda", True)
    assert get_torch_device().type == "cuda"


def test_get_torch_device_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "has_cuda", False)
    monkeypatch.setattr(torch, "has_mps", True)
    assert get_torch_device().type == "mps"


def test_get_torch_device_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "has_cuda", False)
    monkeypatch.setattr(torch, "has_mps", False)
    assert get_torch_device().type == "cpu"
