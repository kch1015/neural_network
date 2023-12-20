import numpy as np

from .optimizer import _Optimizer


class Adam(_Optimizer):
    def __init__(self, params: list, lr: int, momentum=(0.9, 0.999)):
        super().__init__(params, lr)

        self.fig1, self.fig2 = momentum
        self.m = [np.zeros_like(param) for param in self.params]
        self.v = [np.zeros_like(param) for param in self.params]

    def step(self, c=1e-15) -> None:
        for i, param in enumerate(self.params):
            self.m[i] = self.m[i] * self.fig1 + (1 - self.fig1) * param.grad
            self.v[i] = self.v[i] * self.fig2 + (1 - self.fig2) * param.grad * param.grad
            param.data -= self.lr * self.m[i] / (np.sqrt(self.v[i]) + c)
