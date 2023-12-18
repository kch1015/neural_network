import numpy as np


class _Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.params = params

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        pass

