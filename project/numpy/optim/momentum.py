import numpy as np

from .optimizer import _Optimizer


class Momentum(_Optimizer):
    """
    지금까지의 변화율을 저장하여 관성으로 움직여
    local optima에 빠지는 경우를 줄이고 학습 속도를 높이는 최적화 기법이다.
    """
    def __init__(self, params:list , lr: float, momentum=0.9):
        """
        :param params: 갱신할 가중치
        :param lr: 학습률
        :param momentum: 모멘텀 계수, 기존 관성을 모멘텀 계수만큼 유지한다.
        """
        super().__init__(params, lr)
        self.v = [np.zeros_like(param) for param in self.params]
        self.momentum = momentum

    def step(self) -> None:
        for i, param in enumerate(self.params):
            # 새 관성 = 모멘텀 계수 * 기존 관성 - 학습률 * 변화량
            # 가중치 += 관성
            self.v[i] = self.momentum * self.v[i] - self.lr * param.grad
            param += self.v[i]
