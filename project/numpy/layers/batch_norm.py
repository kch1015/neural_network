import numpy as np

from project.numpy.utils.parameter import Parameter


class BatchNorm:
    """
    이전 계층 출력의 평균과 표준편차를 정규화하여
    학습 속도를 높이고, 가중치 초깃값의 영향을 줄이는 최적화 기법
    """
    def __init__(self, num_features: int, momentum=0.9) -> None:
        """
        가중치와 편향을 입력의 요소 수만큼 1, 0으로 초기화한다.
        :param num_features: 입력의 요소 수
        :param momentum: 관성 계수
        """
        self.num_features = num_features
        self.momentum = momentum

        self.weight = Parameter(np.ones(num_features))  # gamma
        self.bias = Parameter(np.zeros(num_features))  # beta

        self.weight.grad = None  # gamma의 변화량
        self.bias.grad = None  # beta의 변화량

        self.running_mean = np.zeros(num_features)  # 이동평균
        self.running_var = np.zeros(num_features)  # 이동분산

        self.batch_size = None
        self.xc = None  # 입력 - 평균
        self.xn = None  # 정규화된 입력
        self.std = None  # 입력의 표준편차

        self.is_training = True  # 학습 여부

    def forward(self, x: np.ndarray, epsilon=1e-15) -> np.ndarray:
        """
        학습 중이면 입력(미니 배치) 내에서 정규화, 관성 계수에 따라 이동 평균과 이동 분산을 갱신하고
        테스트 중이면 계산한 이동 평균과 이동 분산으로 입력을 정규화
        :param x: 입력
        :param epsilon: 0이 되지 않도록 더하는 작은 수
        :return: 배치 정규화 결과
        """
        self.batch_size = x.shape[0]

        if self.is_training:
            mean = np.mean(x, axis=0)
            xc = x - mean
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + epsilon)
            xn = xc / std

            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + epsilon))

        out = self.weight * xn + self.bias

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        가중치, 편향의 변화량을 갱신하고 입력에 대한 변화량을 반환
        :param dout: 배치 정규화 결과에 대한 변화량
        :return: 입력에 대한 변화량
        """
        dxn = self.weight * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std

        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmean = np.sum(dxc, axis=0)

        dx = dxc - dmean / self.batch_size

        self.weight.grad = np.sum(dout * self.xn, axis=0)
        self.bias.grad = np.sum(dout, axis=0)

        return dx
