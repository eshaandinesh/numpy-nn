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
    if tp + fp == 0:
        return 0
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
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def macro_precision(predictions, targets, num_classes):
    """
    predictions: (N,)
    targets: (N,)
    num_classes: int
    returns: float - average precision across all classes
    """
    total = []
    for i in range(num_classes):
        total.append(precision(predictions, targets, i))
    return np.mean(total)

def macro_recall(predictions, targets, num_classes):
    """
    predictions: (N,)
    targets: (N,)
    num_classes: int
    returns: float - average recall across all classes
    """
    total = []
    for i in range(num_classes):
        total.append(recall(predictions, targets, i))
    return np.mean(total)