import torch


def get_torch_device() -> torch.device:
    """
    Returns the torch device to use for inference.
    """
    if torch.has_cuda:
        return torch.device("cuda")

    if torch.has_mps:
        return torch.device("mps")

    return torch.device("cpu")
