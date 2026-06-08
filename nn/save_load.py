import numpy as np

def save_model(model, filepath):
    weights_dict = dict()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'dW'):
            weights_dict[i] = {'W': layer.W, 'b': layer.b}
        if hasattr(layer, 'dgamma'):
            weights_dict[f"{i}_bn"] = {
                'gamma': layer.gamma,
                'beta': layer.beta,
                'running_mean': layer.running_mean,
                'running_var': layer.running_var
            }
    np.save(filepath, weights_dict)

def load_model(model, filepath):
    weights_dict = np.load(filepath, allow_pickle=True).item()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'dW'):
            layer.W = weights_dict[i]["W"]
            layer.b = weights_dict[i]["b"]
        if hasattr(layer, 'dgamma'):
            layer.gamma = weights_dict[f"{i}_bn"]['gamma']
            layer.beta = weights_dict[f"{i}_bn"]['beta']
            layer.running_mean = weights_dict[f"{i}_bn"]['running_mean']
            layer.running_var = weights_dict[f"{i}_bn"]['running_var']

    return model