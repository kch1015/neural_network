from layers import *


class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.affine1 = Affine(input_size, hidden_size)
        self.relu1 = Relu()

        self.affine2 = Affine(hidden_size, output_size)

    def forward(self, x):
        out = self.affine1.forward(x)
        out = self.relu1.forward(out)

        out = self.affine2.forward(out)

        return out