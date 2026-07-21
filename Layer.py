import numpy as np

class Layer:
    def __init__(self, number_neurons, number_weights : 0):
        self.number_neurons = number_neurons
        self.weights = np.array(number_weights)
        # self.bias = 
        print