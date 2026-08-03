import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from activations import sigmoid, softmax


def evaluate(x_valid, y_pred):
    x_valid = x_valid['diagnosis']
    precision = precision_score(x_valid, y_pred, average='micro', zero_division=1)
    print(precision)


def output_layer(x_valid, data):
    prob = softmax(data)
    prob = np.where((prob[:,0] < prob[:,1]), 'B', 'M')
    evaluate(x_valid, prob)
    return


def forward_propagation(weights_bias, x_valid: pd.DataFrame ):
    for i in range(len(weights_bias['weights'])-1):
        if i == 0:
            x = x_valid.drop(columns=['diagnosis']).to_numpy()
            # print(x)
        else:
            x = temp
        temp = np.empty((len(x_valid),0))
        for neuron_index in range(len(weights_bias['weights'][i])):
            z = np.dot(x,weights_bias['weights'][i][neuron_index]) + weights_bias['bias'][i][neuron_index]
            sig = sigmoid(z)
            temp = np.column_stack((temp, sig))
            # print("hey")
    x = temp
    temp = np.empty((len(x_valid),0))
    for neuron_index in range(2):
        z = np.dot(x,weights_bias['weights'][-1][neuron_index]) + weights_bias['bias'][-1][neuron_index]
        temp = np.column_stack((temp, z))
    output_layer(x_valid, temp)


def open_json():
    with open('weights_bias.json', 'r') as f:
        data = json.load(f)
    return data


def measure_precision(validation_dataset : pd.DataFrame):
    weights_bias = open_json()
    forward_propagation(weights_bias, validation_dataset)
