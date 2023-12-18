import unittest

import numpy as np

from project.numpy.layers import Relu


class ReluTest(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(3, 5)
        self.relu = Relu()

    def test_forward(self):
        out1 = np.copy(self.x)
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                if self.x[i, j] <= 0:
                    out1[i, j] = 0

        out2 = self.relu.forward(self.x)

        # print(out1)
        # print(out2)
        self.assertTrue(np.array_equal(out1, out2))

    def test_backward(self):
        dout = np.random.randn(*self.x.shape)

        dx1 = np.copy(dout)
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                if self.x[i, j] <= 0:
                    dx1[i, j] = 0

        out = self.relu.forward(self.x)
        dx2 = self.relu.backward(dout)

        # print(dx1)
        # print(dx2)
        self.assertTrue(np.array_equal(dx1, dx2))
