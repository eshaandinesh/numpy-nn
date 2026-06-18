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
    
class Sigmoid:
    def __init__(self):
        self._output = None

    def forward(self, x):
        """
        x: any shape
        returns: same shape as x
        """
        self._output =  1 / (1 + np.exp(-x))
        return self._output

    def backward(self, dout):
        """
        dout: same shape as x
        returns: same shape as x
        """
        return dout * self._output * (1 - self._output)
    
class Tanh:
    def __init__(self):
        self._output = None

    def forward(self, x):
        """
        x: any shape
        returns: same shape as x
        """
        self._output = np.tanh(x)
        return self._output
        

    def backward(self, dout):
        """
        dout: any shape
        returns: same shape as x
        """
        return dout * (1 - self._output ** 2)