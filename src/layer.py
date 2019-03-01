import copy
from neuron import *


class Layer:

    def __init__(self, size, prevSize, neuron):
        """
        Creates a layer of 'size' neurons and adds a bias neuron at the end.

        :param size: number of neurons
        :param prevSize: number of neurons of previous layer
        :param neuron: type of neurons of the layer
        """
        # neurons' list
        self.neurons = []
        # add hidden neurons
        for i in range(size):
            self.neurons.append(copy.deepcopy(neuron))
            if not isinstance(self.neurons[-1], InputNeuron):
                neuron.weights_init(prevSize+1)
        # add bias neuron
        self.neurons.append(BiasNeuron())

    def getOutput(self):
        """
        returns neurons output
        :return:
        """
        return [neuron.getOutput() for neuron in self.neurons]

    def dump_weights(self, file_output):
        """
        Prints weights to a file, call dump_weights on each neuron
        :param file_output: file to print the weights to
        :return:
        """
        for neuron in self.neurons[:-1]:  # exclude bias
            neuron.dump_weights(file_output)
