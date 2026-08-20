import numpy as np


class Layer:
    """Store a neural-network layer's parameters and optimizer state."""

    def __init__(self, number_neurons: int, number_samples, number_weights=0):
        """Initialize parameters for a fully connected layer.

        Args:
            number_neurons: Number of neurons in the layer.
            number_samples: Number of samples for activation and error buffers.
            number_weights: Number of inputs connected to each neuron.
        """
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

    def copy_layer(self, layer):
        """Replace this layer's parameters and optimizer state with another's."""
        self.bias = layer.bias
        self.weights = layer.weights
        self.momentum_w = layer.momentum_w
        self.momentum_b = layer.momentum_b
        self.RMSProp_w = layer.RMSProp_w
        self.RMSProp_b = layer.RMSProp_b
        self.adam_t = layer.adam_t
        return
