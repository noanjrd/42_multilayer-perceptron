import numpy as np

class Layer:
    def __init__(self, number_neurons: int, number_weights = 0):
        self.number_neurons = number_neurons
        self.errors_respect_weights = np.random.rand(number_neurons, number_weights)
        self.errors_respect_bias = np.random.rand(number_neurons, number_weights)
        self.weights = np.random.rand(number_neurons, number_weights)
        self.bias = np.zeros(number_neurons)
        self.activations = None
        # print(self.weights, self.bias)
        # self.bias = 
        # print()

    def __str__(self):
        return f"Number of neurons: {self.number_neurons}, Number of weights per neurons : {len(self.weights[0])}"