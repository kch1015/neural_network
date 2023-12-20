import numpy as np

from .optimizer import _Optimizer


class Momentum(_Optimizer):
    def __init__(self, params:list , lr: int, momentum=0.9):
        super().__init__(params, lr)
        self.v = [np.zeros_like(param) for param in self.params]
        self.momentum = momentum

    def step(self) -> None:
        for i, param in enumerate(self.params):
            self.v[i] = self.momentum * self.v[i] - self.lr * param.grad
            param.data += self.v[i]
