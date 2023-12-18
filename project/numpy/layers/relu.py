import numpy as np


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.mask * dout
