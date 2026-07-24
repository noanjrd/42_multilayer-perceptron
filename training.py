import pandas as pd
import numpy as np
import argparse
from Layer import Layer

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
        default=1.0,
        help="Learning rate value"
        )
    # print(parser)
    return parser.parse_args()

# def compute_loss():

# def update_neuron():
def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))

def rounds(args, train_data: pd.DataFrame, layer: Layer, x):
    diagnoses = ['M', 'B']
    m = len(x)
    # print(x)
    lambda_reg = 0.1 # How do i decide the right value
    y = (train_data['diagnosis'] == 'M').astype(int).to_numpy()
    res = np.empty((len(x),0))
    for neuron_index in range(layer.number_neurons):
        w = layer.weights[neuron_index]
        b = layer.bias[neuron_index]
        for _ in range(args.epochs):
            indexes = np.random.permutation(len(x))
            x_shuffled = x[indexes]
            y_shuffled = y[indexes]
            for batch_index in range(0, m, args.batch_size):
                x_batch = x_shuffled[batch_index:batch_index+args.batch_size]
                y_batch = y_shuffled[batch_index:batch_index+args.batch_size]
                # print(layer.weights[neuron_index])
                z = np.dot(x_batch, w) + b
                # print(z)
                y_pred = sigmoid(z)
                # print(y_pred)
                errors = y_pred - y_batch
                derivative = np.dot(errors, x_batch) / len(x_batch)

                w = w * (1 - args.learning_rate * (lambda_reg / len(x_batch))) -  args.learning_rate * derivative
                b -= args.learning_rate * errors.mean()
        layer.weights[neuron_index] = w
        layer.bias[neuron_index] = b
        # print(w)
        x_final = np.dot(x, w) + b
        sigmoid_final = sigmoid(x_final)
        # print(sigmoid_final)
        res = np.column_stack((res, sigmoid_final))

    return res

def forward_propagation(args, layers, data):
    training_dataset = pd.read_csv("training_dataset.csv")
    for index, layer in enumerate(layers):
        if index == 0:
            x = training_dataset.select_dtypes(include='number').to_numpy()
        else:
            x = new_data
        new_data = rounds(args, training_dataset, layer, x)
        # print(layer.weights)
        print(new_data)
        break
        print(index)

def start_training(args):
    layers = [Layer(args.layers[i], args.layers[i-1]) for i in range(1,len(args.layers))]
    layers = [Layer(args.layers[0], 31)] + layers
    print(len(layers))
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