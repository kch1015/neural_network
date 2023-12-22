import numpy as np


class Parameter(np.ndarray):
    """
    PyTorch의 Tensor는 backward 호출 시 자동 미분을 수행하고,
    Tensor의 grad 속성에 변화율을 저장한다. Numpy ndarray에도 Tensor처럼
    grad 값을 가질 수 있도록 ndarray를 상속한 Parameter class를 만들었다.
    """
    def __new__(cls, input_array: np.ndarray):
        """
        Parameter를 input_array 값으로 초기화한다.
        :param input_array: 초기화할 배열 값
        """
        parameter = np.asarray(input_array).view(cls)
        parameter.grad = None
        return parameter

    def __array_finalize__(self, parameter):
        if parameter is None:
            return
        self.grad = getattr(parameter, "grad", None)
