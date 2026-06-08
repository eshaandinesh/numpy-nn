# NumPy Neural Network from Scratch

Building a deep learning framework using only NumPy, no PyTorch or TensorFlow.

## Implemented
- `nn/layers.py` — Linear, Conv2D, MaxPool2D, Flatten
- `nn/activations.py` — ReLU
- `nn/losses.py` — CrossEntropyLoss
- `nn/optimizers.py` — SGD, Adam
- `nn/model.py` — Sequential
- `data/loader.py` — MNIST loader via Keras
- MLP (784 → 128 → 64 → 10) — 97.82% test accuracy on MNIST
- CNN (Conv2D → MaxPool2D → Conv2D → MaxPool2D → Flatten → Linear) — 99.14% test accuracy on MNIST

## Next
- Gradient check to verify Conv2D backprop
- Predict function for given image
- Plot the loss curve
