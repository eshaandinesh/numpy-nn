import numpy as np

def save_model(model, filepath):
    weights_dict = dict()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'dW'):
            weights_dict[i] = {'W': layer.W, 'b': layer.b}
    np.save(filepath, weights_dict)

def load_model(model, filepath):
    weights_dict = np.load(filepath, allow_pickle=True).item()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'dW'):
            layer.W = weights_dict[i]["W"]
            layer.b = weights_dict[i]["b"]

    return model