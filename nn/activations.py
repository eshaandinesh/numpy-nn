import numpy as np

class ReLU:
    def __init__(self):
        self._input = None

    def forward(self, x):
        self._input = x
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * (self._input > 0)