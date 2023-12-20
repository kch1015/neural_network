from .optimizer import _Optimizer


class SGD(_Optimizer):
    def __init__(self, params: list, lr: float):
        super().__init__(params, lr)

    def step(self) -> None:
        for param in self.params:
            param.data -= param.grad * self.lr
