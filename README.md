# NumPy Neural Network from Scratch

Building a deep learning framework using only NumPy, no PyTorch or TensorFlow.

## Implemented
- `nn/layers.py` — Linear, Conv2D, MaxPool2D, Flatten
- `nn/activations.py` — ReLU
- `nn/losses.py` — CrossEntropyLoss
- `nn/optimizers.py` — SGD, Adam
- `nn/model.py` — Sequential
- `data/loader.py` — MNIST loader via Keras
- `nn/save_load.py` — save/load model weights
- `data/loader.py` — MNIST loader via Keras
- `train.py` — MLP training, 97.82% test accuracy
- `train_cnn.py` — CNN training, 99.14% test accuracy, loss curve
- `predict.py` — load saved model and run predictions
- `grad_check.py` — numerical gradient check, Linear 8.88e-12, Conv2D 2.24e-11

- MLP (784 → 128 → 64 → 10) — 97.82% test accuracy on MNIST
- CNN (Conv2D → MaxPool2D → Conv2D → MaxPool2D → Flatten → Linear) — 99.14% test accuracy on MNIST

## Next
- LR scheduler
