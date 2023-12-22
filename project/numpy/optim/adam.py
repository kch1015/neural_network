import numpy as np

from .optimizer import _Optimizer


class Adam(_Optimizer):
    """
    RMSProp과 Momentum을 합친 최적화 기법
    """
    def __init__(self, params: list, lr: float, moment=(0.9, 0.999)):
        """
        :param params: 갱신할 가중치
        :param lr:
        :param moment: 1, 2차 적률 계수
        """
        super().__init__(params, lr)
        self.fig1, self.fig2 = moment
        self.m = [np.zeros_like(param) for param in self.params]
        self.v = [np.zeros_like(param) for param in self.params]

    def step(self, c=1e-15) -> None:
        for i, param in enumerate(self.params):
            # 1차 적률 = 1차 적률 계수 * 1차 적률 + (1 - 1차 적률 계수) * 변화량
            self.m[i] = self.m[i] * self.fig1 + (1 - self.fig1) * param.grad
            # 2차 적률 = 2차 적률 계수 * 2차 적률 + (1 - 2차 적률 계수) * 변화량^2
            self.v[i] = self.v[i] * self.fig2 + (1 - self.fig2) * param.grad * param.grad
            # 가중치 -= 학습률 * 1차 적률 / 2차 적률^(1/2)
            param -= self.lr * self.m[i] / (np.sqrt(self.v[i]) + c)
