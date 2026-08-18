import json
import pandas as pd
import numpy as np
import sys
from utils import sigmoid, softmax


def output_layer(data):
    prob = softmax(data)
    prob = np.where((prob[:, 0] < prob[:, 1]), 'B', 'M')
    np.savetxt("predictions_result.csv", prob, fmt="%s")
    return


def forward_propagation(weights_bias, x_valid: pd.DataFrame):
    temp = None
    for i in range(len(weights_bias['weights'])-1):
        if i == 0:
            try:
                x = x_valid.drop(columns=['diagnosis']).to_numpy()
            except Exception:
                x = x_valid.to_numpy()
        else:
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
    with open('weights_bias.json', 'r') as f:
        data = json.load(f)
    return data


def main():
    argv = sys.argv
    file_name = argv[1]
    dataset = pd.read_csv(file_name)
    weights_bias = open_json()
    forward_propagation(weights_bias, dataset)
    return


if __name__ == "__main__":
    main()
