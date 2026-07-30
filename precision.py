import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score

def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def evaluate(y_pred):
    y_test = pd.read_csv("training_dataset.csv")[['diagnosis']]
    precision = precision_score(y_test, y_pred, average='micro', zero_division=1)
    print(precision)

def output_layer(data):
    prob = softmax(data)
    indexes = ['M', 'B']
    # print(soft)
    print(prob)
    prob = np.where((prob[:,0] < prob[:,1]), 'B', 'M')
    print(prob)
    evaluate(prob)
    return

def go(data: pd.DataFrame):
    prediction_data = pd.read_csv("prediction_dataset.csv")
    for i in range(len(data['weights'])-1):
        if i == 0:
            x = prediction_data.to_numpy()
            # print(x)
        else:
            x = temp
        temp = np.empty((len(prediction_data),0))
        for neuron_index in range(len(data['weights'][i])):
            z = np.dot(x,data['weights'][i][neuron_index]) + data['bias'][i][neuron_index]
            sig = sigmoid(z)
            temp = np.column_stack((temp, sig))
            # print("hey")
    x = temp
    temp = np.empty((len(prediction_data),0))
    for neuron_index in range(2):
        z = np.dot(x,data['weights'][-1][neuron_index]) + data['bias'][-1][neuron_index]
        temp = np.column_stack((temp, z))
    output_layer(temp)




    return

def open_json():
    with open('weights_bias.json', 'r') as f:
        data = json.load(f)
    return data

def main():
    data = open_json()
    go(data)
    # print(data)
    return

if __name__ == "__main__":
    main()