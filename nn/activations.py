import numpy as np

class ReLU:
    def __init__(self):
        self._input = None

    def forward(self, x):
        """
        x: (batch_size, any shape)
        returns: (same shape as x)
        """
        self._input = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        dout: (same shape as x)
        returns dx: (same shape as x)
        """
        return dout * (self._input > 0)