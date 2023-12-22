import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.batch_size = None

    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        입력에 Softmax 활성화 함수를 적용하고 교차 엔트로피 손실을 계산해 반환
        :param x: 입력
        :param t: 정답 레이블
        :return: 교차 엔트로피 손실
        """
        self.y = self.softmax(x)
        self.t = t
        loss = self.cross_entropy(self.y, t)
        return loss

    def softmax(self, x: np.ndarray) -> np.ndarray:
        # overflow 방지를 위해 최댓값을 뺌
        max_vals = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_vals)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy(self, y: np.ndarray, t: np.ndarray, c=1e-15) -> np.ndarray:
        self.batch_size = self.y.shape[0]
        # batch_size로 나눠 loss가 batch_size에 영향을 받지 않도록 함
        return -np.sum(t * np.log(y + c)) / self.batch_size

    def backward(self) -> np.ndarray:
        """
        Softmax와 Cross-Entropy의 변화량을 같이 계산하면 y - t로 단순화된다.
        :return: 입력에 대한 손실 변화량
        """
        dx = (self.y - self.t) / self.batch_size
        return dx
