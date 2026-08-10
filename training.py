import pandas as pd
import numpy as np
from Layer import Layer
from TrainingData import TrainingData
from arguments import parse_args
from stats import compute_stats, display_stats
import copy
from utils import save_to_json, sigmoid, softmax


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


def compute_softmax_final_output(training_data: TrainingData,x_train, x_valid, is_training : bool):
    layers = training_data.layers
    table_z_train = np.empty((len(training_data.training_dataset),0))
    table_z_valid = np.empty((len(training_data.validation_dataset),0))
    for neuron_index, diagnos in enumerate(['M', 'B']):
        z_train = np.dot(x_train, training_data.layers[-1].weights[neuron_index]) + layers[-1].bias[neuron_index]
        z_valid = np.dot(x_valid, layers[-1].weights[neuron_index]) + layers[-1].bias[neuron_index]
        table_z_train = np.column_stack((table_z_train, z_train))
        table_z_valid = np.column_stack((table_z_valid, z_valid))
        
    softmax_outcome_train = softmax(table_z_train)
    if not is_training:
        softmax_outcome_valid = softmax(table_z_valid)
        compute_stats(training_data, softmax_outcome_train, softmax_outcome_valid)
    return softmax_outcome_train


def final_output(training_data: TrainingData, x_train, x_valid, is_training):
    layers = training_data.layers
    training_dataset = training_data.training_dataset
    args = training_data.args

    weights_temp = np.empty((0, layers[-1].weights.shape[1]))
    bias_temp = np.empty(0)
    softmax_outcome_train = compute_softmax_final_output(training_data, x_train, x_valid, is_training)

    for neuron_index, diagnos in enumerate(['M', 'B']):
        y = (training_dataset['diagnosis'] == diagnos).astype(int).to_numpy()

        errors = softmax_outcome_train[:,neuron_index] - y  # derivative of BCE 

        derivative_of_weights = errors @ layers[-2].activations.T  / training_dataset.shape[0]
        derivative_of_bias = errors.mean()

        layers[-1].errors[neuron_index] = errors
        weights_temp = np.vstack((weights_temp, derivative_of_weights))
        bias_temp = np.append(bias_temp, derivative_of_bias)

    backpropagation(training_data, len(layers)-2)
    layers[-1].weights -= args.learning_rate * weights_temp
    layers[-1].bias -= args.learning_rate * bias_temp


def compute_activations(layer: Layer, x, is_training: bool):
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


def pass_forward(training_data: TrainingData, is_training: bool):
    layers = training_data.layers
    training_dataset = training_data.training_dataset
    validation_dataset = training_data.validation_dataset

    x_train = None
    x_valid = None
    for index, layer in enumerate(layers):
        if index == len(layers) - 1:
            break
        if index == 0:
            x_train = training_dataset.select_dtypes(include='number').to_numpy()
            x_valid = validation_dataset.select_dtypes(include='number').to_numpy()
        x_train = compute_activations(layer, x_train, is_training)
        x_valid = compute_activations(layer, x_valid, False)

    final_output(training_data, x_train, x_valid, is_training)


def start_training(args):
    x_train, x_valid = pd.read_csv(args.dataset[0]), pd.read_csv(args.dataset[1])
    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_valid.shape}")

    m = len(x_train)
    layers = [Layer(args.layers[i], m, args.layers[i-1]) for i in range(1,len(args.layers))]
    layers = [Layer(args.layers[0], m, 30)] + layers + [Layer(2, m, args.layers[-1])]
    training_data = TrainingData(x_train, x_valid,args, layers)

    training_dataset_copy = training_data.training_dataset.copy()
    validation_dataset_copy = training_data.validation_dataset.copy()

    for _ in range(args.epochs):
        shuffled = training_dataset_copy.sample(frac=1).reset_index(drop=True)
        if training_data.early_stop:
            break
        for index in range(0,shuffled.shape[0],args.batch_size):
            end = min(shuffled.shape[0], index+args.batch_size)
            layers_copy = copy.deepcopy(training_data.layers)
            for layer in training_data.layers:
                layer.errors = layer.errors[:,index:end]
                layer.activations = layer.activations[:,index:end]
            training_data.training_dataset = shuffled[index:end]
            training_data.validation_dataset = shuffled[index:end]

            pass_forward(training_data, True)

            for index, layer in enumerate(training_data.layers):
                layers_copy[index].bias = layer.bias
                layers_copy[index].weights = layer.weights
            training_data.layers = layers_copy
        training_data.training_dataset = training_dataset_copy
        training_data.validation_dataset = validation_dataset_copy
        pass_forward(training_data, False)

    save_to_json(training_data.best_layers)
    display_stats(training_data)


def main():
    try:
        args = parse_args()
        layers = args.layers
        print(layers)
        start_training(args)
    except KeyboardInterrupt:
        print("Program interrupted")
        exit(1)
    return


if __name__ == "__main__":
    main()