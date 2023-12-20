class Backward:
    def __init__(self, model):
        self.affine1 = model.affine1
        self.batchNorm1 = model.batchNorm1
        self.relu1 = model.relu1

        self.affine2 = model.affine2

    def backward(self, dout):
        dout = self.affine2.backward(dout)

        dout = self.relu1.backward(dout)
        dout = self.batchNorm1.backward(dout)
        self.affine1.backward(dout)
