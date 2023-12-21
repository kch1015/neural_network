from abc import abstractmethod


class _Optimizer:
    def __init__(self, params: list, lr: float):
        self.lr = lr
        self.params = params

    @abstractmethod
    def step(self) -> None:
        pass

