from data.loader import load_mnist
from nn.layers import Linear, Conv2D, MaxPool2D, Flatten
from nn.activations import ReLU
from nn.model import Sequential
from nn.save_load import load_model

import numpy as np

# load data
_, _, x_test, y_test = load_mnist()

x_test = x_test.reshape(-1, 28, 28, 1)

# define model
model = Sequential([
    Conv2D(in_channels=1, num_filters=32, filter_size=3),
    ReLU(),
    MaxPool2D(pool_size=2),
    Conv2D(in_channels=32, num_filters=64, filter_size=3),
    ReLU(),
    MaxPool2D(pool_size=2),
    Flatten(),
    Linear(64 * 5 * 5, 128),
    ReLU(),
    Linear(128, 10)
])

model = load_model(model, 'cnn_mnist.npy')
model.eval()

def predict(model, x):
    '''
    x: single image, shape (28, 28, 1)
    reshape to (1, 28, 28, 1) to add batch dimension
    '''
    x = x.reshape(1, 28, 28, 1)
    logits = model.forward(x)
    return np.argmax(logits)

for i in range(5):
    pred = predict(model, x_test[i])
    print(f"Predicted: {pred}, Actual: {y_test[i]}")