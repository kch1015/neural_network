import numpy as np

from .optimizer import _Optimizer


class Adagrad(_Optimizer):
    """
    현재까지 움직인 정도(변화량의 제곱)를 저장하여
    목적지에 가까워질수록 보폭을 줄이는 최적화 기법
    """
    def __init__(self, params: list, lr: float):
        """
        :param params: 갱신할 가중치
        :param lr: 학습률
        """
        super().__init__(params, lr)
        self.h = [np.zeros_like(param) for param in self.params]

    def step(self) -> None:
        for i, param in enumerate(self.params):
            # (1 / 보폭) += 변화량^2
            self.h[i] += param.grad * param.grad
            # 가중치 -= 학습률 * 보폭^(1/2) * 변화량
            param -= self.lr * param.grad / (np.sqrt(self.h[i]) + 1e-15)
