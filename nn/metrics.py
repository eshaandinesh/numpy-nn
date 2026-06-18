import numpy as np

def accuracy(predictions, targets):
    """
    predictions: (N,) - predicted class indices
    targets: (N,) - true class indices
    returns: float between 0 and 1
    """
    return np.mean(predictions == targets)

def precision(predictions, targets, positive_class):
    """
    predictions: (N,) - predicted class indices
    targets: (N,) - true class indices
    positive_class: int - which class is "positive"
    returns: float between 0 and 1
    """
    tp = np.sum((predictions == positive_class) & (targets == positive_class))
    fp = np.sum((predictions == positive_class) & (targets != positive_class))
    return tp / (tp + fp)

def recall(predictions, targets, positive_class):
    """
    predictions: (N,) - predicted class indices
    targets: (N,) - true class indices
    positive_class: int - which class is "positive"
    returns: float between 0 and 1
    """
    tp = np.sum((predictions == positive_class) & (targets == positive_class))
    fn = np.sum((predictions != positive_class) & (targets == positive_class))
    return tp / (tp + fn)