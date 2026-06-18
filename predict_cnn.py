from data.loader import load_mnist
from nn.layers import Linear, Conv2D, MaxPool2D, Flatten, BatchNorm, Dropout
from nn.activations import ReLU
from nn.model import Sequential
from nn.save_load import load_model

import numpy as np
import matplotlib.pyplot as plt

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
    BatchNorm(128),
    ReLU(),
    Dropout(p=0.3),
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

# visualize predictions on a grid of random test samples
indices = np.random.choice(len(x_test), 10, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(15, 7))

for ax, idx in zip(axes.flatten(), indices):
    img = x_test[idx]
    pred = predict(model, img)
    actual = y_test[idx]

    ax.imshow(img.reshape(28, 28), cmap='gray')
    color = 'darkgreen' if pred == actual else 'red'
    ax.set_title(f"Pred: {pred}\nTrue: {actual}", color=color, fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')