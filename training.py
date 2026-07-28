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

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def final_output(args, train_data : pd.DataFrame, layers: list[Layer], x):
    res = softmax(x)
    print(res)
    for neuron_index, diagnos in enumerate(['M', 'B']):
        y = (train_data['diagnosis'] == diagnos).astype(int).to_numpy()
        print(y, y.shape)
        errors = res[:,0] - y
        derivative_of_weights = np.dot(errors, layers[-2].activations) / 588
        derivative_of_bias = errors.mean()
        layers[-1].weights[neuron_index] -= args.learning_rate * derivative_of_weights
        layers[-1].bias[neuron_index] -= args.learning_rate * derivative_of_bias
        layers[-1].errors_respect_weights[neuron_index] = derivative_of_weights
        layers[-1].errors_respect_bias[neuron_index] = derivative_of_bias
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
    final_output(args, training_dataset,layers ,new_data)

def back_gradient_descent(args,current_layer_index, layers:list[Layer]):
    for neuron_index in range(layers[current_layer_index].number_neurons):
        errors = np.dot(layers[current_layer_index + 1].errors_respect_weights)

    return

def backpropagation(aregs, layers:list[Layer]):
    for layer_index in range(len(layers)-2,0,-1):
        back_gradient_descent(args, layer_index, layers)
    return


def start_training(args):
    layers = [Layer(args.layers[i], args.layers[i-1]) for i in range(1,len(args.layers))]
    layers = [Layer(args.layers[0], 31)] + layers + [Layer(2, args.layers[-1])]
    [print(layer) for layer in layers]
    data = pd.read_csv('training_dataset.csv')
    forward_propagation(args, layers, data)
    backpropagation(args, layers)
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