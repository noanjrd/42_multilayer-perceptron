import pandas as pd
import numpy as np
from Layer import Layer
import json
from arguments import parse_args
from activations import sigmoid, softmax


def backpropagation(args, training_data:pd.DataFrame, current_layer_index:int, layers:list[Layer]):
    current_layer = layers[current_layer_index]
    next_layer = layers[current_layer_index + 1]
    x = training_data.select_dtypes(include='number').to_numpy()
    weights_temp = np.empty((0, current_layer.weights.shape[1]))
    bias_temp = np.empty(0)

    for neuron_index in range(current_layer.number_neurons):
    
        delta =  np.sum(next_layer.errors * next_layer.weights[:, neuron_index][:, None], axis=0)
        activations = current_layer.activations[neuron_index]
        delta *= (activations * (1 - activations))
        current_layer.errors[neuron_index] = delta

        if current_layer_index != 0:
            gradient_weights = (layers[current_layer_index - 1].activations @ delta) / len(training_data)
        else:
            gradient_weights = x.T @ delta / len(training_data)
        gradient_bias = np.mean(delta)

        weights_temp = np.vstack((weights_temp, gradient_weights))
        bias_temp = np.append(bias_temp, gradient_bias)

    if current_layer_index > 0:
        backpropagation(args, training_data, current_layer_index - 1, layers)

    # print(args.learning_rate)
    current_layer.weights -= args.learning_rate * weights_temp
    current_layer.bias -= args.learning_rate * bias_temp
    return


def final_output(args, train_data : pd.DataFrame, layers: list[Layer], x):
    tab = np.empty((len(train_data),0))
    for neuron_index, diagnos in enumerate(['M', 'B']):
        z = np.dot(x, layers[-1].weights[neuron_index]) + layers[-1].bias[neuron_index]
        tab = np.column_stack((tab, z))
        
    res = softmax(tab)
    weights_temp = np.empty((0, layers[-1].weights.shape[1]))
    bias_temp = np.empty(0)
    for neuron_index, diagnos in enumerate(['M', 'B']):
        y = (train_data['diagnosis'] == diagnos).astype(int).to_numpy()

        errors = res[:,neuron_index] - y  # derivative of BCE

        derivative_of_weights = errors @ layers[-2].activations.T  / len(train_data)
        derivative_of_bias = errors.mean()
        
        layers[-1].errors[neuron_index] = errors
        weights_temp = np.vstack((weights_temp, derivative_of_weights))
        bias_temp = np.append(bias_temp, derivative_of_bias)

    backpropagation(args, train_data, len(layers)-2, layers)
    layers[-1].weights -= args.learning_rate * weights_temp
    layers[-1].bias -= args.learning_rate * bias_temp


def rounds(layer: Layer, x):
    res = np.empty((len(x),0))
    for neuron_index in range(layer.number_neurons):
        w = layer.weights[neuron_index]
        b = layer.bias[neuron_index]
        z = np.dot(x, w) + b

        y_prediction = sigmoid(z)
        layer.activations[neuron_index] = y_prediction
        res = np.column_stack((res, y_prediction))
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

    final_output(args, training_data,layers ,new_data)


def start_training(args):
    data = pd.read_csv('training_dataset.csv')
    m = len(data)
    layers = [Layer(args.layers[i], m, args.layers[i-1]) for i in range(1,len(args.layers))]
    layers = [Layer(args.layers[0], m, 30)] + layers + [Layer(2, m, args.layers[-1])]
    [print(layer) for layer in layers]
    for _ in range(args.epochs):
        forward_propagation(args,data , layers )

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
    print(layers)
    start_training(args)
    return


if __name__ == "__main__":
    main()