import numpy as np


class SquaredError:
    def __init__(self):
        pass

    def value(self, target, output_net):
        return np.sum(np.square((target - output_net))) / 2

    def derivative(self, target, output_net):
        return target - output_net
