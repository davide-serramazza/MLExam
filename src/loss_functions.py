import numpy as np


class SquaredError:
    def __init__(self,t):
        self.type = t

    def value(self, target, output_net):
        return np.sum(np.square((target - output_net))) / 2

    def misClassification(self, target, output_net):
        if self.type == "s":
            return np.sum(np.square((target - np.round(output_net))))
        if self.type == "t":
            return np.sum(np.square(target-np.sign(output_net)[0]))/4

    def derivative(self, target, output_net):
        return target - output_net
