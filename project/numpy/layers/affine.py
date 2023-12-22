import numpy as np

from project.numpy.utils.parameter import Parameter


class Affine:
    """
    Fully-Connected 계층
    """
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        입력, 출력 크기를 받아 가중치를 초기화
        :param in_features: 입력 수
        :param out_features: 출력 수
        :param bias: 편향 여부
        """
        self.input_size = in_features
        self.output_size = out_features

        self.weight = Parameter(np.sqrt(2.0 / self.input_size) * np.random.randn(in_features, out_features))
        # self.weight = Parameter(np.random.randn(in_features, out_features))
        # self.weight = Parameter(np.ones((in_features, out_features)))

        self.bias = None
        if bias:
            self.bias = Parameter(np.sqrt(2.0 / self.input_size) * np.random.randn(1, out_features))
            # self.bias = Parameter(np.random.randn(1, out_features))
            # self.bias = Parameter(np.ones((1, out_features)))

        self.x = None

    def get_parameters(self):
        return self.weight, self.bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        입력에 가중치를 곱해 편향을 더한 순전파 결과를 반환한다.
        :param x: 입력
        :return: 순전파 결과
        """
        self.x = x

        out = np.dot(x, self.weight)
        if self.bias is not None:
            out += self.bias

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        가중치, 편향의 변화량을 계산하고 입력의 변화량은 반환한다.
        :param dout: 순전파 결과에 대한 변화량
        :return: 입력에 대한 변화량
        """
        dx = np.dot(dout, self.weight.T)

        self.weight.grad = np.dot(self.x.T, dout)
        self.bias.grad = np.sum(dout, axis=0)

        return dx
