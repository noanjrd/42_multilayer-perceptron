import json
import pandas as pd
import numpy as np
import sys
from utils import sigmoid, softmax


def output_layer(data):
    """Convert output logits to diagnosis labels and write the predictions CSV."""
    prob = softmax(data)
    prob = np.where((prob[:, 0] < prob[:, 1]), 'B', 'M')
    np.savetxt("predictions_result.csv", prob, fmt="%s")
    return


def forward_propagation(weights_bias, x_valid: pd.DataFrame) -> None:
    """Run inference using serialized weights and write predicted labels.

    The input may include a ``diagnosis`` column, which is ignored when present.
    Hidden layers use sigmoid activations and the two-neuron output layer uses
    softmax classification.
    """
    temp: np.ndarray = x_valid.drop(columns=["diagnosis"], errors="ignore").to_numpy()
    for i in range(len(weights_bias['weights'])-1):
        x = temp
        temp = np.empty((len(x_valid), 0))
        for neuron_index in range(len(weights_bias['weights'][i])):
            z = np.dot(x, weights_bias['weights'][i][neuron_index]) + weights_bias['bias'][i][neuron_index]
            sig = sigmoid(z)
            temp = np.column_stack((temp, sig))
    x = temp
    temp = np.empty((len(x_valid), 0))
    for neuron_index in range(2):
        z = np.dot(x, weights_bias['weights'][-1][neuron_index]) + weights_bias['bias'][-1][neuron_index]
        temp = np.column_stack((temp, z))
    output_layer(temp)


def open_json():
    """Load model weights and biases from ``weights_bias.json``."""
    with open('weights_bias.json', 'r') as f:
        data = json.load(f)
    return data


def main():
    try:
        argv = sys.argv
        assert len(argv) == 2, "Need one argument, the dataset the program will make the predictions on"
        file_name = argv[1]
        dataset = pd.read_csv(file_name)
        weights_bias = open_json()
        forward_propagation(weights_bias, dataset)
    except AssertionError as e:
        print("Error:", e)
        exit(1)
    except Exception:
        print("Error")
        exit(1)


if __name__ == "__main__":
    main()
