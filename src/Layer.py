import Neuron
import numpy as np
import copy
import random
import abc
from Neuron import *


class Layer:
    def __init__(self, size, prevSize, neuron):
        # neurons' list
        self.neurons = []
        # add hidden neurons
        for i in range(size):
            self.neurons.append(copy.deepcopy(neuron))
        # add bias neuron
        self.neurons.append(BiasNeuron())

    def getOutput (self):
        res = []
        for i in range ( len (self.neurons)):
            res.append(self.neurons[i].getOutput())
        return res

    # TODO add output_file as a parameter
    def dump_weights(self):
        # print weights to a file
        # call serialize on each neuron
        for neuron in self.neurons[:-1]:  # exclude bias
            neuron.dump_weights()
