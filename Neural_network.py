import numpy as np
from Layer import *


class Network:
    def __init__(self, architecture):
        # input layer
        self.layers = []
        tmp = InputNeuron()
        l = Layer(architecture[0], 0, tmp)
        self.layers.append(l)
        # hidden layers
        for i in range(1, len(architecture) - 1):
            tmp = SigmoidNeuron()
            l = Layer(architecture[i], architecture[i - 1], tmp)
            self.layers.append(l)
        # output layers
        tmp = OutputNeuron()
        size = len(architecture) - 1
        l = Layer(architecture[size], architecture[size - 1], tmp)
        self.layers.append(l)

    def train(self):
        # fit the data
        pass

    def predict(self):
        # predict target variable
        pass

    def serialize(self):
        # dump neural network weights to file
        # calls serialize on each layer
        pass
