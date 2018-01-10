import numpy as np
import sys

#superclass abstract Neuron
class Neuron(object):
    def __init__(self,len_weights):
        self.output = 0.0
        self.weights = np.random.uniform(low=-0.7, high=0.7, size=len_weights)

    def weights_init(self,len_weights):
        self.weights = np.random.uniform(low=-0.7, high=0.7, size=len_weights)

    def activation_function(self, x):
        pass

    def activation_function_derivative(self):
        pass

    def getOutput(self):
        return self.output

    def dump_weights(self, file_output):
        # prints weights to a file
        if file_output == sys.stdout:
            print self.weights
        else:
            np.save(file_output, self.weights)


# subclasses
class InputNeuron(Neuron):

    def activation_function(self, x):
        self.output = x
        return self.output

    def activation_function_derivative(self):
        return 1


class SigmoidNeuron(Neuron):

    def activation_function(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def activation_function_derivative(self):
        return self.output * (1 - self.output);


class ReLuNeuron(Neuron):

    def activation_function(self, x):
        self.output = np.max(0, x)
        return self.output

    def activation_function_derivative(self):
        return 1 if self.output > 0 else 0


class TanHNeuron(Neuron):

    def activation_function(self, x):
        sigmoid = lambda y: (1 / (1 + np.exp(-y)))
        self.output = 2 * sigmoid(2*x) - 1

    def activation_function_derivative(self):
        return 1 - (self.output * self.output)


class BiasNeuron(Neuron):
    def __init__(self):
        self.output = 1.0

    def activation_function(self, x):
        return self.output

    def activation_function_derivative(self):
        return 0.0


class OutputNeuron(Neuron):

    def activation_function(self, x):
        self.output = x
        return self.output

    def activation_function_derivative(self):
        return 1
