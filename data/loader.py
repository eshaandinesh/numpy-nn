import numpy as np

def load_mnist():
    from tensorflow.keras.datasets import mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # flatten from (28, 28) to (784,)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # normalize to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test