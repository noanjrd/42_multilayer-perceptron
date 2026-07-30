import pandas as pd
import numpy as np
import argparse
from Layer import Layer
import json

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
    tab = np.empty((len(train_data),0))
    for neuron_index, diagnos in enumerate(['M', 'B']):
        z = np.dot(x, layers[-1].weights[neuron_index]) + layers[-1].bias[neuron_index]
        tab = np.column_stack((tab, z))
        
    # print(tab)
    res = softmax(tab)
    for neuron_index, diagnos in enumerate(['M', 'B']):
        y = (train_data['diagnosis'] == diagnos).astype(int).to_numpy( )
        # print(y, y.shape, layers[-2].activations)
        errors = res[:,neuron_index] - y
        derivative_of_weights = errors @ layers[-2].activations.T  / len(train_data)
        # print(derivative_of_weights)
        # print(derivative_of_weights)
        derivative_of_bias = errors.mean()
        # print(layers[-1].weights)
        layers[-1].weights[neuron_index] -= args.learning_rate * derivative_of_weights
        layers[-1].bias[neuron_index] -= args.learning_rate * derivative_of_bias
        layers[-1].errors[neuron_index] = errors
    return res


def rounds(layer: Layer, x):
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
        layer.activations[neuron_index] = y_pred
        res = np.column_stack((res, y_pred))
    return res

def forward_propagation(args, training_data, layers):
    new_data = None
    for index, layer in enumerate(layers):
        if index == len(layers) - 1:
            break
        if index == 0:
            x = training_data.select_dtypes(include='number').to_numpy()
        else:
            x = new_data
        new_data = rounds(layer, x)
        # print(layer.weights)
        # print(new_data)
        # break
        # print(index)
    # print(new_data, new_data.shape)
    final_output(args, training_data,layers ,new_data)

def back_gradient_descent(args, training_data, current_layer_index, layers:list[Layer]):
    # print(layers[current_layer_index + 1].errors_respect_weights[neuron_index])
    current_layer = layers[current_layer_index]
    next_layer = layers[current_layer_index + 1]
    x = training_data.select_dtypes(include='number').to_numpy()
    for neuron_index in range(current_layer.number_neurons):
        # for next_layer_neuron_index in range(layers[current_layer_index + 1].number_neurons):
            # print(layers[current_layer_index + 1].weights[neuron_index].shape)
        delta =  np.sum(next_layer.errors * next_layer.weights[:, neuron_index][:, None], axis=0)
        # errors_bias =  np.sum(layers[current_layer_index + 1].errors_respect_bias * layers[current_layer_index + 1].bias[:, neuron_index][:, None], axis=0)
        # print(errors, "h")
        # print(layers[current_layer_index].activations[neuron_index], "act")
        activation = current_layer.activations[neuron_index]
        delta *= (activation * (1 - activation))
        if current_layer_index != 0:
            gradient_weights = (layers[current_layer_index - 1].activations @ delta) / len(training_data)
        else:
            gradient_weights = x.T @ delta / len(training_data)
        gradient_bias = np.mean(delta)
        # errors_weights = np.dot(layers[current_layer_index-1].activations[neuron_index], errors_weights) / 568
        # print(errors_weights, errors_weights.shape)
        
        current_layer.weights[neuron_index] -= args.learning_rate * gradient_weights
        current_layer.bias[neuron_index] -= args.learning_rate * gradient_bias
        current_layer.errors[neuron_index] = delta  

    return

def backpropagation(args, training_data, layers:list[Layer]):
    for layer_index in range(len(layers)-2,-1,-1):
        back_gradient_descent(args, training_data ,layer_index, layers)
    return


def start_training(args):
    data = pd.read_csv('training_dataset.csv')
    m = len(data)
    layers = [Layer(args.layers[i], m, args.layers[i-1]) for i in range(1,len(args.layers))]
    layers = [Layer(args.layers[0], m, 31)] + layers + [Layer(2, m, args.layers[-1])]
    [print(layer) for layer in layers]
    for _ in range(args.epochs):
        forward_propagation(args,data , layers )
        backpropagation(args, data, layers)
    # for i in range(len(layers)):
    #     print(layers[i].weights)
    save_to_json(layers)
    return


def save_to_json(layers: list[Layer]):
    data = {'weights': [], 'bias' : []}
    for layer in layers:
        data['weights'].append(layer.weights.tolist())
        data['bias'].append(layer.bias.tolist())
    with open('weights_bias.json', 'w') as f:
        json.dump(data, f, indent=4)
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