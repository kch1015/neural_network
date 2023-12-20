import numpy as np

from .optimizer import _Optimizer


class RMSprop(_Optimizer):
    def __init__(self, params: list, lr: int, weight_decay=0.99):
        super().__init__(params, lr)
        self.h = self.h = [np.zeros_like(param) for param in self.params]
        self.weight_decay = weight_decay

    def step(self) -> None:
        for i, param in enumerate(self.params):
            self.h[i] = self.h[i] * self.weight_decay + (1 - self.weight_decay) * param.grad * param.grad
            param.data -= self.lr * param.grad / (np.sqrt(self.h[i]) + 1e-15)
