import numpy as np
from nn.layers import Linear, Conv2D, Flatten
from nn.activations import ReLU
from nn.losses import CrossEntropyLoss
from nn.model import Sequential

def gradient_check():
    np.random.seed(42)
    
    # small network and small input for speed
    model = Sequential([
        Linear(4, 8),
        ReLU(),
        Linear(8, 3)
    ])
    loss_fn = CrossEntropyLoss()
    
    x = np.random.randn(2, 4)
    y = np.array([0, 2])
    
    # get analytic gradients
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)
    dout = loss_fn.backward()
    model.backward(dout)
    
    # numerical gradient check on first layer weights
    h = 1e-5
    W = model.layers[0].W
    analytic_grad = model.layers[0].dW
    numerical_grad = np.zeros_like(W)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            original = W[i][j]
            W[i][j] = original + h
            loss_plus = loss_fn.forward(model.forward(x), y)
            W[i, j] = original - h
            loss_minus = loss_fn.forward(model.forward(x), y)

            W[i, j] = original
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * h)
    
    diff = np.max(np.abs(analytic_grad - numerical_grad))
    print(f"Max difference: {diff:.2e}")
    print("PASSED" if diff < 1e-4 else "FAILED")

def gradient_check_conv():
    np.random.seed(42)
    
    # small everything for speed
    model = Sequential([
        Conv2D(in_channels=1, num_filters=4, filter_size=3),
        ReLU(),
        Flatten(),
        Linear(4 * 6 * 6, 3)
    ])
    loss_fn = CrossEntropyLoss()
    
    x = np.random.randn(2, 8, 8, 1)
    y = np.array([0, 2])
    
    # analytic gradients
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)
    dout = loss_fn.backward()
    model.backward(dout)
    
    # numerical gradient check on Conv2D filters
    h = 1e-5
    W = model.layers[0].W
    analytic_grad = model.layers[0].dW
    numerical_grad = np.zeros_like(W)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                for l in range(W.shape[3]):
                    original = W[i][j][k][l]
                    W[i][j][k][l] = original + h
                    loss_plus = loss_fn.forward(model.forward(x), y)
                    W[i, j][k][l] = original - h
                    loss_minus = loss_fn.forward(model.forward(x), y)

                    W[i, j][k][l] = original
                    numerical_grad[i, j][k][l] = (loss_plus - loss_minus) / (2 * h)
    
    diff = np.max(np.abs(analytic_grad - numerical_grad))
    print(f"Conv2D Max difference: {diff:.2e}")
    print("PASSED" if diff < 1e-4 else "FAILED")

gradient_check()

gradient_check_conv()