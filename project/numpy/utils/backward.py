class Backward:
    """
    PyTorch의 Tensor.backward()의 역할을 수행하는 클래스이다.
    PyTorch의 자동 미분과는 달리 model의 모든 계층의 backward()를 직접 반대 방향으로 호출한다.
    """
    def __init__(self, model):
        self.affine1 = model.affine1
        self.batchNorm1 = model.batchNorm1
        self.relu1 = model.relu1

        self.affine2 = model.affine2

    def __call__(self, dout):
        dout = self.affine2.backward(dout)

        dout = self.relu1.backward(dout)
        dout = self.batchNorm1.backward(dout)
        self.affine1.backward(dout)
