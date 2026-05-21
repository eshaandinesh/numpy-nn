import numpy as np
from nn.layers import Linear
from nn.activations import ReLU

np.random.seed(42)

layer = Linear(3, 2)
relu = ReLU()

x = np.array([[1.0, 2.0, 3.0]])

out = relu.forward(layer.forward(x))
print("Output shape:", out.shape)

dout = np.ones((1, 2))
dx = layer.backward(relu.backward(dout))
print("dW shape:", layer.dW.shape)
print("dx shape:", dx.shape)