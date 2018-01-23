import numpy as np


class SquaredError:
    def __init__(self, t):
        self.type = t

    def value(self, target, output_net, weights, regularization=0):
        """
        Computes value of the squared error (data error + regularization error).

        :param target: target of the training example
        :param output_net: neural network output
        :param weights: vector of all weights of the neural network
        :param regularization: regularization strength
        :return:
        """
        # data error
        difference = target - output_net
        data_error = np.sum(np.square(difference)) / 2
        # regularization error
        regularization_error = regularization * np.sum(np.square(weights)) / 2
        return data_error + regularization_error

    def misClassification(self, target, output_net):
        """
        Computes misclassfication error. Adjusts computation according to the type of neurons of the output layer.
        (assuming only 1 output)
        This is ignored for the regression task.

        :param target: target of the training example
        :param output_net: neural network output for the training example
        :return:
        """
        if self.type == "sigmoid":
            return np.sum(np.square((target - np.round(output_net))))
        if self.type == "tangentH":
            return np.sum(np.square(target - np.sign(output_net)[0]))/4

    def derivative(self, target, output_net):
        """
        Computes derivative of the error function.

        :param target: vector containing the target of the training example
        :param output_net: vector containing the neural network output for the training example
        :return: value of the derivative of the error for the training example
        """
        return target - output_net
