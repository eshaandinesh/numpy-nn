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

## Next
- Train CNN on MNIST (target: 99%+)
- Gradient check to verify Conv2D backprop
