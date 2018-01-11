import numpy as np


class SquaredError:
    def __init__(self, t):
        self.type = t

    def value(self, target, output_net, weights, regularization=0):
        # data error
        difference = target - output_net
        data_error = np.sum(np.square(difference)) / 2
        # regularization error
        weights = np.concatenate(weights)
        regularization_error = regularization * np.sum(np.square(weights)) / 2
        return data_error + regularization_error

    def misClassification(self, target, output_net):
        if self.type == "sigmoid":
            return np.sum(np.square((target - np.round(output_net))))
        if self.type == "tangentH":
            return np.sum(np.square(target-np.sign(output_net)[0]))/4

    def derivative(self, target, output_net):
        return target - output_net
