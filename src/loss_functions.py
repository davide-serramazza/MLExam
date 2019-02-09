import numpy as np
from numpy.linalg import norm
from numpy import square, sum, round, sign


class SquaredError:

    def __init__(self, t=""):
        self.type = t

    def value(self, target, output_net, weights=[], regularization=0):
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
        data_error = sum(square(difference))
        regu_error = regularization * sum(square(weights))
        return data_error + regu_error

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
            return sum(square((target - round(output_net))))
        if self.type == "tangentH":
            return sum(square(target - sign(output_net)[0]))/4
        else:
            return np.Inf

    def derivative(self, target, output_net):
        """
        Computes derivative of the error function.

        :param target: vector containing the target of the training example
        :param output_net: vector containing the neural network output for the training example
        :return: value of the derivative of the error for the training example
        """
        return 2 * (output_net - target)


class EuclideanError:
    def value(self, target, output_net, weights=[], regularization=0):
        """
        computes Euclidean error between targets and output
        :param target: target vector
        :param output_net: output vector
        :param weights: network weights
        :param regularization: regularization strength
        :return:
        """
        # data error
        data_error = norm(output_net - target)
        regu_error = regularization * sum(square(weights))
        return data_error + regu_error

    def derivative(self, target, output_net):
        """
        computes derivative of Squared Error
        :param target:
        :param output_net:
        :return:
        """
        difference = output_net - target
        return difference / norm(difference)

    def misClassification(self, target, output_net):
        # ignore
        return 0
