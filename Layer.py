import numpy as np


class Layer:
    def __init__(self, number_neurons: int, number_samples, number_weights=0):
        self.number_neurons = number_neurons
        self.errors = np.random.rand(number_neurons, number_samples)
        self.weights = np.random.uniform(-1, 1, size=(number_neurons, number_weights))
        self.bias = np.zeros(number_neurons)
        self.momentum_w = np.zeros_like(self.weights)
        self.momentum_b = np.zeros_like(self.bias)
        self.RMSProp_w = np.zeros_like(self.weights)
        self.RMSProp_b = np.zeros_like(self.bias)
        self.adam_t = 0
        self.activations = np.zeros((number_neurons, number_samples))

    def __str__(self):
        return f"Number of neurons: {self.number_neurons}, Number of weights per neurons : {len(self.weights[0])}"
