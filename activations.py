import numpy as np


def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)