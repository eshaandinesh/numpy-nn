import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self._probs = None
        self._labels = None

    def forward(self, logits, labels):
        """
        logits: (batch_size, num_classes) - raw scores
        labels: (batch_size,) - correct class indices
        returns: scalar loss
        """
        self._labels = labels
        
        x = logits - np.max(logits, axis=1, keepdims=True)
        self._probs = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        correct_probs = self._probs[np.arange(len(labels)), labels]
        loss = - np.mean(np.log(correct_probs))

        return loss 

    def backward(self):
        """
        returns dx: (batch_size, num_classes)
        """
        one_hot = np.zeros_like(self._probs)
        one_hot[np.arange(len(self._labels)), self._labels] = 1 
        dx = self._probs - one_hot
        dx /= len(self._labels)
        return dx
    
class MSELoss:
    def __init__(self):
        self._predictions = None
        self._targets = None

    def forward(self, predictions, targets):
        """
        predictions: (batch_size, output_dim)
        targets: (batch_size, output_dim)
        returns: scalar loss
        """
        self._predictions = predictions
        self._targets = targets
        loss = np.mean((self._predictions - self._targets) ** 2)
        return loss

    def backward(self):
        """
        returns dx: (batch_size, output_dim)
        """
        dx = 2 * (self._predictions - self._targets) / len(self._targets)
        return dx