import numpy as np
import sys


# superclass abstract Neuron
class Neuron(object):
    """ Abstract Neuron class.
    Its subclasses implements the activation function and its derivative.

    Parameters
    ----------
    len_weights : float
        number of parameters attached to the neuron

    Attributes
    ----------
    output : float
        neuron's output after computing the activation_function
    weights : vector
        neuron's parameters

    """
    def __init__(self, len_weights):
        """
        Create and initialize the neuron.

        len_weights: number of weights
        """
        self.output = 0.0
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=len_weights)

    def weights_init(self, len_weights):
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=len_weights)

    def activation_function(self, x):
        """ activation function of the neuron.

        Parameters
        ----------
        x : float
            dot product between the neuron's weights and its input
        Returns
        -------
        g(x): float
        """
        pass

    def activation_function_derivative(self):
        """
        derivative of the activation function
        """
        pass

    def getOutput(self):
        return self.output

    def dump_weights(self, file_output):
        """
        Save the weights of the neuron to a 'file_output'

        file_output: file to print the weights to
        """
        if file_output == sys.stdout:
            print self.weights
        else:
            np.save(file_output, self.weights)


# concrete subclasses
class InputNeuron(Neuron):
    """ placeholder used to map the input into the network.
    """
    def activation_function(self, x):
        self.output = x
        return self.output

    def activation_function_derivative(self):
        return 1


class SigmoidNeuron(Neuron):
    """ Implements the logistic activation function and its derivative
    """
    def activation_function(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def activation_function_derivative(self):
        return self.output * (1 - self.output);


class ReLuNeuron(Neuron):
    """ Implements the Rectifier Linear Unit activation function and its derivative
    """
    def activation_function(self, x):
        self.output = max(0, x)
        return self.output

    def activation_function_derivative(self):
        return 1 if self.output > 0 else 0


class TanHNeuron(Neuron):
    """ Implements the Iperbolic tangent activation function and its derivative
    """
    def activation_function(self, x):
        self.output = np.tanh(x)
        return self.output

    def activation_function_derivative(self):
        return 1.0 - self.output**2

class BiasNeuron(Neuron):
    """ Implements the bias neuron.
    """
    def __init__(self):
        self.output = 1.0

    def activation_function(self, x):
        return self.output

    def activation_function_derivative(self):
        return 0.0

class LinearNeuron(Neuron):
    """ Implements the linear/identity activation function and its derivative
    """
    def activation_function(self, x):
        self.output = x
        return self.output

    def activation_function_derivative(self):
        return 1
