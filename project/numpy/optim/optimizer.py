import numpy as np


class _Optimizer:
    def __init__(self, params, lr):
        self.lr = lr
        self.params = params

    def step(self) -> None:
        pass

