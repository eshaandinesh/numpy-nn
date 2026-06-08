import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, layers):
        """
        layers: list of layer objects
        updates W and b in-place for all layers with dW
        """
        for layer in layers:
            if hasattr(layer, 'dW'):
                layer.W = layer.W - self.lr * layer.dW
                layer.b = layer.b - self.lr * layer.db


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep
        self.m = {}  # first moment for each layer
        self.v = {}  # second moment for each layer

    def step(self, layers):
        """
        layers: list of layer objects
        updates W and b in-place for all layers with dW
        """
        self.t += 1
        for i, layer in enumerate(layers):
            if hasattr(layer, 'dW'):
                # initialize m and v for this layer if first time
                if i not in self.m:
                    self.m[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.v[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

                self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1 - self.beta1) * layer.dW        # smooth the gradient
                self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * layer.db        # smooth the gradient
                self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1 - self.beta2) * layer.dW**2       # track gradient magnitude
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * layer.db**2       # track gradient magnitude

                m_W = self.m[i]['W'] / (1 - self.beta1**self.t)                   # bias correction
                m_b = self.m[i]['b'] / (1 - self.beta1**self.t)                   # bias correction
                v_W = self.v[i]['W'] / (1 - self.beta2**self.t)                   # bias correction
                v_b = self.v[i]['b'] / (1 - self.beta2**self.t)                   # bias correction

                layer.W = layer.W - self.lr * m_W / (np.sqrt(v_W) + self.epsilon)
                layer.b = layer.b - self.lr * m_b / (np.sqrt(v_b) + self.epsilon)