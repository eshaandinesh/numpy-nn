# NumPy Neural Network from Scratch

Building a deep learning framework using only NumPy, no PyTorch or TensorFlow.

## Implemented
- `nn/layers.py` — Linear layer (forward + backprop)
- `nn/activations.py` — ReLU
- `nn/losses.py` — CrossEntropyLoss (softmax + cross entropy, numerically stable)
- `nn/optimizers.py` — SGD
- `nn/model.py` — Sequential container
- `data/loader.py` — MNIST loader via Keras
- MLP (784 → 128 → 64 → 10) — 97.82% test accuracy on MNIST

## Next
- Adam optimizer
- Conv2D + MaxPool2D
- CNN on MNIST (target: 99%+)
