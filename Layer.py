import Neuron
import numpy as np
import copy
import random
import abc
from Neuron import *

class Layer:
    def __init__(self, size, prevSize, neuron):
        # neuron's and weight's list
        self.neurons = []
        self.w = []
        # add hidden neurons
        for i in range(size):
            self.neurons.append(copy.deepcopy(neuron))
        # add bias neuron
        self.neurons.append(BiasNeuron())
        # add weights
        if prevSize != 0:  # if is not an input layer
            for i in range(size):
                self.w.append([])
                for j in range(prevSize):
                    self.w[i].append(random.uniform(-1.0, 1.0))


    def serialize(self):
        # print weights to a file
        # call serialize on each neuron
        pass