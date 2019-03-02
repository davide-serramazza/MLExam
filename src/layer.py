import copy
from neuron import *


class Layer:
    """ A Layer is a list of Neuron

    Parameters
    ----------
    size : int
        number of neurons in the layer
    prevSize : int
        number of neurons in the previous layer
    neuron : Neuron
        type of Neuron to be used in this layer
    Attributes
    ----------
    neurons : Neuron
        type of neurons to be used in this layer
    """

    def __init__(self, size, prevSize, neuron):
        """
        Creates a layer of 'size' neurons and adds a bias neuron at the end.

        size: number of neurons
        prevSize: number of neurons of previous layer
        neuron: type of neurons of the layer
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
        return: float
            neurons output
        """
        return [neuron.getOutput() for neuron in self.neurons]

    def dump_weights(self, file_output):
        """
        Prints weights to a file, call dump_weights on each neuron

        file_output: file to print the weights to
        """
        for neuron in self.neurons[:-1]:  # exclude bias
            neuron.dump_weights(file_output)
