import numpy as np

#superclass abstract Neuron
class Neuron(object):
    def __init__(self):
        self.output = 0.0

    def activation_function(self, x):
        pass

    def activation_function_derivate(self):
        pass

    def getResult(self):
        return self.output

    def serialize(self):
        # prints weights to a file
        pass

# subclasses
class InputNeuron(Neuron):
    def activation_function(self, x):
        self.output = x
        return self.output

    def activation_function_derivate(self):
        return 1


class SigmoidNeuron(Neuron):
    def activation_function(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def activation_function_derivate(self):
        return self.output * (1 - self.output);


class BiasNeuron(Neuron):
    def __init__(self):
        self.output = 1.0

    def activation_function(self, x):
        return self.output

    def activation_function_derivate(self):
        return 0.0


class OutputNeuron(Neuron):
    def activation_function(self, x):
        self.output = x
        return self.output
