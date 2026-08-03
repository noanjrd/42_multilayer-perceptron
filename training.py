import pandas as pd
import numpy as np
from Layer import Layer
from TrainingData import TrainingData
import json
from arguments import parse_args
from activations import sigmoid, softmax
from validation import  measure_precision


def save_to_json(layers: list[Layer]):
    data = {'weights': [], 'bias' : []}
    for layer in layers:
        data['weights'].append(layer.weights.tolist())
        data['bias'].append(layer.bias.tolist())
    with open('weights_bias.json', 'w') as f:
        json.dump(data, f, indent=4)
    return 


def backpropagation(training_data:TrainingData, current_layer_index:int):
    layers = training_data.layers
    args = training_data.args
    training_dataset =training_data.training_dataset

    current_layer = layers[current_layer_index]
    next_layer = layers[current_layer_index + 1]

    x = training_dataset.select_dtypes(include='number').to_numpy()

    weights_temp = np.empty((0, current_layer.weights.shape[1]))
    bias_temp = np.empty(0)

    for neuron_index in range(current_layer.number_neurons):
        delta =  np.sum(next_layer.errors * next_layer.weights[:, neuron_index][:, None], axis=0)
        activations = current_layer.activations[neuron_index]
        delta *= (activations * (1 - activations))
        current_layer.errors[neuron_index] = delta

        if current_layer_index != 0:
            gradient_weights = (layers[current_layer_index - 1].activations @ delta) / len(training_dataset)
        else:
            gradient_weights = x.T @ delta / len(training_dataset)
        gradient_bias = np.mean(delta)

        weights_temp = np.vstack((weights_temp, gradient_weights))
        bias_temp = np.append(bias_temp, gradient_bias)

    if current_layer_index > 0:
        backpropagation(training_data, current_layer_index - 1)

    current_layer.weights -= args.learning_rate * weights_temp
    current_layer.bias -= args.learning_rate * bias_temp
    return

def calculate_loss(training_data: TrainingData,  softmax_prob_train, softmax_prob_valid):
    training_dataset = training_data.training_dataset
    validation_dataset = training_data.validation_dataset
    epochs_stats = training_data.epoch_stats

    y_train = (training_dataset['diagnosis'] == 'M').astype(int).to_numpy()
    y_valid = (validation_dataset['diagnosis'] == 'M').astype(int).to_numpy()

    p_train = softmax_prob_train
    p_valid = softmax_prob_valid
    loss_train = -np.mean(y_train * np.log(p_train[:, 0]) + (1 - y_train) * np.log(p_train[:, 1]))
    loss_valid = -np.mean(y_valid * np.log(p_valid[:, 0]) + (1 - y_valid) * np.log(p_valid[:, 1]))
    # print(loss)
    epochs_stats.append((loss_train, loss_valid))
    return


def final_output(training_data: TrainingData, x_train, x_valid):
    layers = training_data.layers
    training_dataset = training_data.training_dataset
    validation_dataset = training_data.validation_dataset
    args = training_data.args

    table_z_train = np.empty((len(training_dataset),0))
    table_z_valid = np.empty((len(validation_dataset),0))
    for neuron_index, diagnos in enumerate(['M', 'B']):
        z_train = np.dot(x_train, layers[-1].weights[neuron_index]) + layers[-1].bias[neuron_index]
        z_valid = np.dot(x_valid, layers[-1].weights[neuron_index]) + layers[-1].bias[neuron_index]
        table_z_train = np.column_stack((table_z_train, z_train))
        table_z_valid = np.column_stack((table_z_valid, z_valid))
        
    softmax_prob_train = softmax(table_z_train)
    softmax_prob_valid = softmax(table_z_valid)
    calculate_loss(training_data, softmax_prob_train, softmax_prob_valid)

    weights_temp = np.empty((0, layers[-1].weights.shape[1]))
    bias_temp = np.empty(0)
    for neuron_index, diagnos in enumerate(['M', 'B']):
        y = (training_dataset['diagnosis'] == diagnos).astype(int).to_numpy()

        errors = softmax_prob_train[:,neuron_index] - y  # derivative of BCE

        derivative_of_weights = errors @ layers[-2].activations.T  / len(training_dataset)
        derivative_of_bias = errors.mean()
        
        layers[-1].errors[neuron_index] = errors
        weights_temp = np.vstack((weights_temp, derivative_of_weights))
        bias_temp = np.append(bias_temp, derivative_of_bias)

    backpropagation(training_data, len(layers)-2)
    layers[-1].weights -= args.learning_rate * weights_temp
    layers[-1].bias -= args.learning_rate * bias_temp


def calculate_activations(layer: Layer, x, is_training: bool):
    res = np.empty((len(x),0))
    for neuron_index in range(layer.number_neurons):
        w = layer.weights[neuron_index]
        b = layer.bias[neuron_index]
        z = np.dot(x, w) + b

        y_prediction = sigmoid(z)
        if is_training:
            layer.activations[neuron_index] = y_prediction
        res = np.column_stack((res, y_prediction))
    return res


def pass_forward(training_data: TrainingData):
    layers = training_data.layers
    training_dataset = training_data.training_dataset
    validation_dataset = training_data.validation_dataset

    new_input_train = None
    new_input_valid = None
    for index, layer in enumerate(layers):
        if index == len(layers) - 1:
            break
        if index == 0:
            x_valid = validation_dataset.select_dtypes(include='number').to_numpy()
            x_train = training_dataset.select_dtypes(include='number').to_numpy()
        else:
            x_valid = new_input_valid
            x_train = new_input_train
        new_input_valid = calculate_activations(layer, x_valid, False)
        new_input_train = calculate_activations(layer, x_train, True)

    final_output(training_data, new_input_train, new_input_valid)


def start_training(args):
    x_train, x_valid = pd.read_csv(args.dataset[0]), pd.read_csv(args.dataset[1])
    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_valid.shape}")

    m = len(x_train)
    layers = [Layer(args.layers[i], m, args.layers[i-1]) for i in range(1,len(args.layers))]
    layers = [Layer(args.layers[0], m, 30)] + layers + [Layer(2, m, args.layers[-1])]
    training_data = TrainingData(x_train, x_valid,args, layers)

    for _ in range(args.epochs):
        pass_forward(training_data)

    print(training_data.epoch_stats)
    measure_precision(x_valid)
    save_to_json(layers)
    return


def main():
    args = parse_args()
    layers = args.layers
    print(layers)
    start_training(args)
    return


if __name__ == "__main__":
    main()