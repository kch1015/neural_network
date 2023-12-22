from .optimizer import _Optimizer


class SGD(_Optimizer):
    """
    미니 배치만큼 Gradient Descent를 적용한다.
    """
    def __init__(self, params: list, lr: float):
        """
        :param params: 갱신할 가중치
        :param lr: 학습률
        """
        super().__init__(params, lr)

    def step(self) -> None:
        for param in self.params:
            # 가중치 -= 학습률 * 변화량
            param -= param.grad * self.lr
