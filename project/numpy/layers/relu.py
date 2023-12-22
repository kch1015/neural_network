import numpy as np


class Relu:
    """
    입력이 0 이하면 0을, 0 초과면 그대로 출력하는 활성화 함수
    """
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        입력에서 0 초과인 인덱스를 저장하고 0 이하인 요소는 0으로 바꿔 반환한다.
        :param x: 입력
        :return: 활성화 결과
        """
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        순전파에서 저장한 마스크를 기반으로 입력이 0 이하였던 요소의 변화량을 0으로 반환한다.
        :param dout: 활성화 결과에 대한 변화량
        :return: 입력에 대한 변화량
        """
        return self.mask * dout
