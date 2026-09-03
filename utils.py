import json
from Layer import Layer
import numpy as np
import pandas as pd

def normalize_dataset(dataset: pd.DataFrame, mins_and_maxs: list[tuple[int,int]] = None) -> None:
    """Min-max normalize feature columns in place, preserving diagnosis labels."""
    new_mins_maxs = []
    i = 0
    for col in dataset.columns:
        if col in ['diagnosis']:
            continue
        if mins_and_maxs is None:
            min = dataset[col].min()
            max = dataset[col].max()
            new_mins_maxs.append((min, max))
        else:
            min = mins_and_maxs[i][0]
            max = mins_and_maxs[i][1]
        dataset[col] = (dataset[col] - min) / (max - min)
        i+=1
        
    # print(new_mins_maxs)
    return new_mins_maxs


def save_to_json(layers: list[Layer], mins_and_maxs) -> None:
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
    with open('scales.json', 'w') as f:
        json.dump(mins_and_maxs, f, indent=4)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Apply the element-wise logistic sigmoid function."""
    return 1 / (1 + (np.exp(-z)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Normalize each row of a two-dimensional logits array into probabilities."""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


columns = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]