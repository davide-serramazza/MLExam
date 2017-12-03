
class SquaredError:
    def __init__(self):
        self.loss = 0
        self.loss_derivative = 0

    def loss(self, target, output_net):
        self.loss = (target - output_net)**2 / 2
        self.loss_derivative = target - output_net
        return self.loss

    def derivative(self, target, output_net):
        return target - output_net
