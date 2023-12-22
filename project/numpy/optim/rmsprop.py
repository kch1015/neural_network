import numpy as np

from .optimizer import _Optimizer


class RMSProp(_Optimizer):
    """
    AdaGrad의 단점을 보완한 최적화 기법
    보폭을 갈수록 줄이되 이전 값의 영향을 AdaGrad보다 더 크게 받아
    보폭이 갑자기 줄어들지 않아 local optima에 빠지는 경우가 줄어든다.
    """
    def __init__(self, params: list, lr: float, weight_decay=0.99):
        """
        :param params: 갱신할 가중치
        :param lr: 학습률
        :param weight_decay: 감쇠율, 이전 보폭이 감쇠율만큼, 새로 영향을 주는 변화량이 (1 - 감쇠율)만큼의 비율을 차지한다.
        """
        super().__init__(params, lr)
        self.h = self.h = [np.zeros_like(param) for param in self.params]
        self.weight_decay = weight_decay

    def step(self) -> None:
        for i, param in enumerate(self.params):
            # (1 / 보폭) = 감쇠율 * (1 / 보폭) + (1 - 감쇠율) * 변화량^2
            self.h[i] = self.h[i] * self.weight_decay + (1 - self.weight_decay) * param.grad * param.grad
            # 가중치 -= 학습률 * 보폭^(1/2) * 변화량
            param -= self.lr * param.grad / (np.sqrt(self.h[i]) + 1e-15)
