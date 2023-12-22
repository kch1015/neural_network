from abc import abstractmethod


class _Optimizer:
    """
    다른 optimizer의 기본 클래스이자 추상 클래스이다.
    """
    def __init__(self, params: list, lr: float):
        """
        :param params: 갱신할 가중치
        :param lr: 학습률
        """
        self.lr = lr
        self.params = params

    @abstractmethod
    def step(self) -> None:
        """
        클래스 상속 시 반드시 오버라이딩이 필요한 함수
        :return:
        """
        pass

