from data.loader import load_mnist
from nn.layers import Linear
from nn.activations import ReLU
from nn.losses import CrossEntropyLoss
from nn.optimizers import Adam
from nn.model import Sequential
from nn.save_load import save_model

import numpy as np
import matplotlib.pyplot as plt

# load data
x_train, y_train, x_test, y_test = load_mnist()

# define model
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
])

loss_fn = CrossEntropyLoss()
optimizer = Adam(learning_rate=0.001)

# training loop
epochs = 10
batch_size = 32

model.train() # doesnt matter rn because no dropout used

loss_history = []

for epoch in range(epochs):
    # shuffle data
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]

    total_loss = 0
    num_batches = len(x_train) // batch_size

    for i in range(num_batches):
        # get batch
        x_batch = x_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_train[i*batch_size:(i+1)*batch_size]

        # forward
        logits = model.forward(x_batch)
        loss = loss_fn.forward(logits, y_batch)

        # backward
        dout = loss_fn.backward()
        model.backward(dout)

        # update weights
        optimizer.step(model.layers)

        total_loss += loss

    loss_history.append(total_loss / num_batches)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/num_batches:.4f}")

epoch_range = range(1, len(loss_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epoch_range, loss_history, label='Training Loss', marker='o')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('mlp_loss_curve.png')

save_model(model, 'mlp_mnist.npy')

print("Testing")

model.eval() # doesnt matter rn because no dropout

logits = model.forward(x_test)
predictions = np.argmax(logits, axis=1)
accuracy = np.mean(predictions == y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")