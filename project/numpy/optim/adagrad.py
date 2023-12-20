import numpy as np

from .optimizer import _Optimizer


class Adagrad(_Optimizer):
    def __init__(self, params: list, lr: int):
        super().__init__(params, lr)
        self.h = [np.zeros_like(param) for param in self.params]

    def step(self) -> None:
        for i, param in enumerate(self.params):
            self.h[i] += param.grad * param.grad
            param.data -= self.lr * param.grad / (np.sqrt(self.h[i]) + 1e-15)
