from data.loader import load_mnist
from nn.layers import Linear
from nn.activations import ReLU
from nn.model import Sequential
from nn.save_load import load_model

import numpy as np

# load data
_, _, x_test, y_test = load_mnist()

# define model
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
])

model = load_model(model, 'mlp_mnist.npy')
model.eval()

def predict(model, x):
    '''
    x: single image, shape (784,)
    '''
    logits = model.forward(x)
    return np.argmax(logits)

for i in range(5):
    pred = predict(model, x_test[i])
    print(f"Predicted: {pred}, Actual: {y_test[i]}")