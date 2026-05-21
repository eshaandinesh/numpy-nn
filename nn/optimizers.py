class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'dW'):
                layer.W = layer.W - self.lr * layer.dW
                layer.b = layer.b - self.lr * layer.db