import numpy as np

from project.numpy.utils.parameter import Parameter

class Affine:
    def __init__(self, in_features: int, out_features: int, bias=True):
        self.input_size = in_features
        self.output_size = out_features

        self.weight = Parameter(np.sqrt(2 / self.input_size) * np.random.randn(in_features, out_features))

        self.bias = None
        if bias:
            self.bias = Parameter(np.sqrt(2 / self.input_size) * np.random.randn(1, out_features))

        self.x = None

    def get_parameters(self):
        return self.weight, self.bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x

        out = np.dot(x, self.weight)
        if self.bias is not None:
            out += self.bias

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = np.dot(dout, self.weight.T)

        self.weight.grad = np.dot(self.x.T, dout)
        self.bias.grad = np.sum(dout, axis=0)

        return dx
