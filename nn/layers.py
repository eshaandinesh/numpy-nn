import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        # He initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros((1, out_features))
        
        self.dW = None
        self.db = None
        self._input = None  # cache for backward pass

    def forward(self, x):
        """
        x: (batch_size, in_features)
        returns: (batch_size, out_features)
        """
        self._input = x
        return x @ self.W + self.b    # y = xW + b

    def backward(self, dout):
        """
        dout: (batch_size, out_features)
        returns dx: (batch_size, in_features)
        """
        self.dW = self._input.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True) 
        return dout @ self.W.T