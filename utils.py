import json
from Layer import Layer
import numpy as np


def save_to_json(layers: list[Layer]) -> None:
    """Serialize layer weights and biases to ``weights_bias.json``."""
    data: dict[str, list[list[float] | list[list[float]]]] = {
        'weights': [],
        'bias': [],
    }
    print("> saving model './weights_bias.json' to disk...")
    for layer in layers:
        data['weights'].append(layer.weights.tolist())
        data['bias'].append(layer.bias.tolist())
    with open('weights_bias.json', 'w') as f:
        json.dump(data, f, indent=4)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Apply the element-wise logistic sigmoid function."""
    return 1 / (1 + (np.exp(-z)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Normalize each row of a two-dimensional logits array into probabilities."""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)
