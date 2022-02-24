import numpy as np
from utils import *


class Neuron:
    def __init__(self, num_weights, bias):
        self.weights = get_random_values(num_weights)
        self.bias = bias

    def get_net(self, inputs):
        net = np.dot(self.weights, inputs) + self.bias
        return threshold_fun(net)

    def train(self, data, goal):
        n_j = 0.1
        for key, value in data.items():
            y = self.get_net(value)
            yl = 1 if int(key) == goal else 0
            for i in range(len(self.weights)):
                xi = value[i]
                self.weights[i] -= n_j * (y - yl) * xi

    def test(self, data):
        r = []
        for key, value in data.items():
            r += [(key, self.get_net(value))]
        return r






