import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.batch_size = None

    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.y = self.softmax(x)
        self.t = t
        loss = self.cross_entropy(self.y, t)
        return loss

    def softmax(self, x: np.ndarray) -> np.ndarray:
        max_vals = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_vals)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy(self, y: np.ndarray, t: np.ndarray, c=1e-15) -> np.ndarray:
        self.batch_size = self.y.shape[0]
        return -np.sum(t * np.log(y + c)) / self.batch_size

    def backward(self):
        dx = (self.y - self.t) / self.batch_size
        return dx
