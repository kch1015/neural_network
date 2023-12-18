import numpy as np


class Parameter(np.ndarray):
    def __new__(cls, input_array, grad=None):
        parameter = np.asarray(input_array).view(cls)
        parameter.grad = grad
        return parameter

    def __array_finalize__(self, parameter):
        if parameter is None:
            return
        self.grad = getattr(parameter, "grad", None)
