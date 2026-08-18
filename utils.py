import json
from Layer import Layer
import numpy as np


def save_to_json(layers: list[Layer]):
    data = {'weights': [], 'bias': []}
    print("saving model './weights_bias.json' to disk...")
    for layer in layers:
        data['weights'].append(layer.weights.tolist())
        data['bias'].append(layer.bias.tolist())
    with open('weights_bias.json', 'w') as f:
        json.dump(data, f, indent=4)
    return


def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)
