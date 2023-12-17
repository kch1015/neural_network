import numpy as np
import torch


def one_hot_encode(label: np.ndarray, size=10) -> torch.Tensor:
    return torch.zeros(size, dtype=torch.float).scatter_(0, torch.tensor(label), value=1)
