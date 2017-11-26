import unittest

from src.Neural_network import *


class TestNeuralNetwork(unittest.TestCase):
    def forwardTest(self):
        arch = [2, 2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, OutputNeuron]
        network = Network(arch, neuronsType, [0.05, 0.1], [0.01, 0.99])

        network.layers[1].neurons[0].weights = [0.15, 0.2, 0.35]
        network.layers[1].neurons[1].weights = [0.25, 0.3, 0.35]
        network.layers[2].neurons[0].weights = [0.4, 0.45, 0.6]
        network.layers[2].neurons[1].weights = [0.5, 0.55, 0.6]
        self.assertEquals(network.forward(), [1.1059, 1.2249])

    def topologyTest(self):
        arch = [2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, ReLuNeuron, OutputNeuron]
        with self.assertRaises(Exception):
            network = Network(arch, neuronsType, [0.05, 0.1], [0.01, 0.99])

        neuronsType = [SigmoidNeuron, ReLuNeuron, OutputNeuron]
        with self.assertRaises(Exception):
            network = Network([2,2,2], neuronsType, [0.05, 0.1], [0.01, 0.99])

        neuronsType = [InputNeuron, OutputNeuron, OutputNeuron]
        with self.assertRaises(Exception):
            network = Network([2, 2, 2], neuronsType, [0.05, 0.1], [0.01, 0.99])

        neuronsType = [SigmoidNeuron, ReLuNeuron, TanHNeuron]
        with self.assertRaises(Exception):
            network = Network([2, 2, 2], neuronsType, [0.05, 0.1], [0.01, 0.99])


if __name__ == '__main__':
    unittest.main()
