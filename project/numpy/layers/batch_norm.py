import numpy as np

from project.numpy.utils.parameter import Parameter


class BatchNorm:
    def __init__(self, num_features: int, momentum=0.9) -> None:
        self.num_features = num_features
        self.momentum = momentum

        self.weight = Parameter(np.ones(num_features))  # gamma,
        self.bias = Parameter(np.zeros(num_features))  # beta

        self.weight.grad = None  # gamma의 기울기
        self.bias.grad = None  # beta의 기울기

        self.running_mean = np.zeros(num_features)  # 이동평균
        self.running_var = np.zeros(num_features)  # 이동분산

        self.batch_size = None  # 배치 사이즈
        self.xc = None  # 입력값에 평균을 뺀 값
        self.xn = None  # 정규화된 값
        self.std = None  # 표준편차

        self.is_training = True  # 학습 중인지 여부

    def forward(self, x: np.ndarray, epsilon=1e-15) -> np.ndarray:
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
