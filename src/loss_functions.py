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
        data_error = np.sum(np.square(difference))
        # regularization error
        regularization_error = regularization * np.sum(np.square(weights))
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
        # TODO return (output - target) so it is the true gradient and then use "- eta" in update_weights, and also
        # TODO cancel the minus in the bfgs compute_gradient()
        return 2 * (output_net - target)


class EuclideanError:
    def __init__(self,normalizer=None):
        self.normalizer = normalizer

    def value(self, target, output_net, weights, regularization=0):
        #denormalize
        if self.normalizer != None:
            den_target = self.normalizer.inverse_transform([target])
            den_output = self.normalizer.inverse_transform([output_net])
            target = den_target[0]
            output_net = den_output[0]

        # data error
        data_error = np.linalg.norm(output_net - target)

    # regularization error
        regularization_error = regularization * np.sum(np.square(weights))
        return data_error + regularization_error

    def derivative(self, target, output_net):
        return 2 * (output_net - target)

    def misClassification(self, target, output_net):
        # ignore
        return 0
