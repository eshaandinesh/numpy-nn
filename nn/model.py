class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        """
        x: input array of any shape
        returns: output after passing through all layers in order
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        """
        dout: gradient from next layer
        returns: gradient after passing through all layers in reverse
        """
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)
        return dout
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False