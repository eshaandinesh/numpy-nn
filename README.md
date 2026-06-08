# NumPy Neural Network from Scratch

Building a deep learning framework using only NumPy, no PyTorch or TensorFlow.

## Implemented
- `nn/layers.py` — Linear, Conv2D, MaxPool2D, Flatten, BatchNorm, Dropout
- `nn/activations.py` — ReLU
- `nn/losses.py` — CrossEntropyLoss
- `nn/optimizers.py` — SGD, Adam, ReduceLROnPlateau
- `nn/model.py` — Sequential (train/eval mode)
- `nn/save_load.py` — save/load model weights
- `data/loader.py` — MNIST loader via Keras
- `train.py` — MLP training, 98.12% test accuracy
- `train_cnn.py` — CNN training, 99.14% test accuracy
- `predict.py` — CNN inference
- `predict_mlp.py` — MLP inference
- `grad_check.py` — gradient verification, Linear 8.88e-12, Conv2D 2.24e-11