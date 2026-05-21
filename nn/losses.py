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