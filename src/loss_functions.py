import numpy as np
from numpy.linalg import norm
from numpy import square, sum, round, sign


class SquaredError:
    """
    Implements Squared Error loss function

    Parameters
    ----------
    t : string
        {"", "sigmoid", "tangentH"} depending on the output layer's activation function
    Attributes
    ----------
    type : string
    """

    def __init__(self, t=""):
        self.type = t

    def value(self, target, output_net, regularization, weights):
        """
        Computes value of the squared error (data error + regularization error).

        target: training pattern's labels
        output_net: neural network's output
        weights: neural network's weights
        regularization: regularization strength

        return:
        loss: data error + regularization error
        data_error: data error only
        """
        data_error = sum(square(output_net - target))
        reg_error = regularization * sum(square(weights))
        loss = data_error + reg_error
        return loss, data_error

    def misClassification(self, target, output_net):
        """
        Computes misclassfication error.
        Adjusts computation according to the type of neurons of the output layer.
        (assuming only 1 output)
        This is ignored for the regression task.

        target: target of the training example
        output_net: neural network output for the training example

        return:
            misClassification error
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

        target: vector containing the target of the training example
        output_net: vector containing the neural network output for the training example

        return:
            value of the derivative of the error for the training example
        """
        return 2 * (output_net - target)


class EuclideanError:
    """ Implements the Euclidean Error loss function"""
    def value(self, target, output_net, regularization, weights):
        """
        computes Euclidean error between targets and output

        target: target vector
        output_net: network output vector
        weights: network weights
        regularization: regularization strength

        return:
            loss: data error + regularization error
            data_error: data error only
        """
        # data error
        data_error = norm(output_net - target)
        reg_error = regularization * sum(square(weights))
        loss = data_error + reg_error
        return loss, data_error

    def derivative(self, target, output_net):
        """
        computes derivative of Euclidean Error

        target:     target vector
        output_net: neural network output vector
        """
        difference = output_net - target
        return difference / norm(difference)

    def misClassification(self, target, output_net):
        # ignore
        return 0
