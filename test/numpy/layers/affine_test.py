import unittest

import numpy as np

from project.numpy.layers import Affine


class AffineTest(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(2, 5)
        self.affine = Affine(5, 3, bias=True)

    def test_forward(self):
        weight = self.affine.weight
        bias = self.affine.bias

        out1 = self.affine.forward(self.x)
        out2 = np.dot(self.x, weight) + bias

        # print(out1)
        # print(out2)
        self.assertTrue(np.array_equal(out1, out2))

    def test_backward(self):
        out = self.affine.forward(self.x)
        dout = np.random.randn(*out.shape)

        weight, bias = self.affine.get_parameters()

        dx1 = self.affine.backward(dout)
        dx2 = np.dot(dout, weight.T)

        dw1 = weight.grad
        dw2 = np.dot(self.x.T, dout)

        db1 = bias.grad
        db2 = np.sum(dout, axis=0)

        self.assertTrue(np.array_equal(dx1, dx2))
        self.assertTrue(np.array_equal(dw1, dw2))
        self.assertTrue(np.array_equal(db1, db2))
