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
            if not isinstance(self.neurons[-1],InputNeuron):
                neuron.weights_init(prevSize+1)
        # add bias neuron
        self.neurons.append(BiasNeuron())

    def getOutput(self):
        res = []
        for i in range ( len (self.neurons)):
            res.append(self.neurons[i].getOutput())
        return res

    def dump_weights(self, file_output):
        # print weights to a file
        # call serialize on each neuron
        for neuron in self.neurons[:-1]:  # exclude bias
            neuron.dump_weights(file_output)
        #print >> file_output  # prints \n at the end of each layer
