import numpy as np


class SquaredError:
    def __init__(self, t,):# regularization):
        self.type = t
        #self.regularization = regularization

    def value(self, target, output_net, regularization=0, weights=[0]):
        return (np.sum(np.square((target - output_net))) +
        regularization * np.multiply(weights, weights)) / 2  # L2-regularization

    def misClassification(self, target, output_net):#, weights):
        if self.type == "s":
            return np.sum(np.square((target - np.round(output_net))))
        if self.type == "t":
            return np.sum(np.square(target-np.sign(output_net)[0]))/4

    def derivative(self, target, output_net):
        return target - output_net
