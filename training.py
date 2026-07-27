import pandas as pd
import numpy as np
import argparse
from Layer import Layer

class_to_index = {
    'malignant' : 0,
    'benign' : 0
}

def parse_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="?", # how many values '?' for 0 or 1, '+' for one or more
        required=False,
        default=20,
        help="Number of training epochs"
        )
    
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        required=False,
        default=[5,4],
        help="Number of hidden layers and their neurons"
        )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        required=False,
        default=10,
        help="Size of the batches"
        )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        nargs="?",
        required=False,
        default=0.01,
        help="Learning rate value"
        )
    # print(parser)
    return parser.parse_args()

# def compute_loss():

# def update_neuron():
def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

def backpropagation():
    return

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def final_output(x):
    res = softmax(x)
    return res


def rounds(args, train_data: pd.DataFrame, layer: Layer, x):
    res = np.empty((len(x),0))
    for neuron_index in range(layer.number_neurons):
        w = layer.weights[neuron_index]
        b = layer.bias[neuron_index]
                # print(layer.weights[neuron_index])
        z = np.dot(x, w) + b
                # print(z)
        y_pred = sigmoid(z)
                # print(y_pred)
        # print(w)
        # print(sigmoid_final)
        layer.activations = y_pred
        res = np.column_stack((res, y_pred))
    return res

def forward_propagation(args, layers, data):
    training_dataset = pd.read_csv("training_dataset.csv")
    new_data = None
    for index, layer in enumerate(layers):
        if index == 0:
            x = training_dataset.select_dtypes(include='number').to_numpy()
        else:
            x = new_data
        new_data = rounds(args, training_dataset, layer, x)
        # print(layer.weights)
        # print(new_data)
        # break
        # print(index)
    # print(new_data, new_data.shape)
    res = softmax(new_data)
    print(res, res.shape)

def start_training(args):
    layers = [Layer(args.layers[i], args.layers[i-1]) for i in range(1,len(args.layers))]
    layers = [Layer(args.layers[0], 31)] + layers + [Layer(2, args.layers[-1])]
    [print(layer) for layer in layers]
    data = pd.read_csv('training_dataset.csv')
    forward_propagation(args, layers, data)
    return

def main():
    args = parse_args()
    layers = args.layers
    # parse_args()
    print(layers)
    start_training(args)
    return

if __name__ == "__main__":
    main()