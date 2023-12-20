import numpy as np

from layers import *


class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.affine1 = Affine(input_size, hidden_size)
        self.batchNorm1 = BatchNorm(hidden_size)
        self.relu1 = Relu()

        self.affine2 = Affine(hidden_size, output_size)

    def __call__(self, x) -> np.ndarray:
        out = self.affine1.forward(x)
        out = self.batchNorm1.forward(out)
        out = self.relu1.forward(out)

        out = self.affine2.forward(out)

        return out

    def parameters(self) -> list:
        return [self.affine1.weight, self.affine1.bias,
                self.batchNorm1.weight, self.batchNorm1.bias,
                self.affine2.weight, self.affine2.bias]

    def train(self):
        self.batchNorm1.is_training = True
        # pass

    def eval(self):
        self.batchNorm1.is_training = False
        # pass
