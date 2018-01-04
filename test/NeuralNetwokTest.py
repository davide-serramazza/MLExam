import unittest

from src.Neural_network import *

# TODO we may use assertAlmostEqual() if numerical issues arise
# each test must start with test_
class TestNeuralNetwork(unittest.TestCase):
    def test_forward(self):
        arch = [2, 2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, SigmoidNeuron]
        data = [0.05, 0.1]
        network = Network(arch, neuronsType)

        self.set_weights(network)

        network.forward(data)
        network_output = (network.output)
        expected_output = [0.75136506955231575, 0.77292846532146253]
        self.assertEquals(network_output, expected_output)

    def test_backward(self):
        arch = [2, 2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, SigmoidNeuron]
        network = Network(arch, neuronsType)

        self.set_weights(network)

        data = [0.05, 0.1]
        target = [0.01, 0.99]
        network.forward(data)
        delta_w, loss_value, _ = network.back_propagation(target=target, eta=0.5)
        network.update_weights(delta_w)

        layers = network.layers
        self.assert_weights(layers)
        self.assertEqual(loss_value.round(9), 0.298371109)  # oracle up to 9 digits

    def test_train(self):
        arch = [2, 2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, SigmoidNeuron]
        network = Network(arch, neuronsType)
        self.set_weights(network)

        data = [[0.05, 0.1]]
        target = [[0.01, 0.99]]
        network.train(data=data, targets=target, epochs=1, learning_rate=0.5, batch_size=1, momentum=0)

        layers = network.layers
        self.assert_weights(layers)

    def assert_weights(self, layers):
        self.assertEqual(layers[1].neurons[0].weights, [0.14978071613276281, 0.19956143226552567, 0.34561432265525649])
        self.assertEqual(layers[1].neurons[1].weights, [0.24975114363236958, 0.29950228726473915, 0.3450228726473914])
        self.assertEqual(layers[2].neurons[0].weights, [0.35891647971788465, 0.4086661860762334, 0.53075071918572148])
        self.assertEqual(layers[2].neurons[1].weights, [0.5113012702387375, 0.56137012110798912, 0.61904911825827813])

    def set_weights(self, network):
        network.layers[1].neurons[0].weights = [0.15, 0.2, 0.35]
        network.layers[1].neurons[1].weights = [0.25, 0.3, 0.35]
        network.layers[2].neurons[0].weights = [0.4, 0.45, 0.6]
        network.layers[2].neurons[1].weights = [0.5, 0.55, 0.6]

    def test_topology(self):
        neuronsType = [InputNeuron, SigmoidNeuron, ReLuNeuron, OutputNeuron]
        with self.assertRaises(Exception):
            Network([2,2], neuronsType)

        neuronsType = [SigmoidNeuron, ReLuNeuron, OutputNeuron]
        with self.assertRaises(Exception):
            Network([2,2,2], neuronsType)

        neuronsType = [InputNeuron, OutputNeuron, OutputNeuron]
        with self.assertRaises(Exception):
            Network([2, 2, 2], neuronsType)

        neuronsType = [SigmoidNeuron, ReLuNeuron, TanHNeuron]
        with self.assertRaises(Exception):
            Network([2, 2, 2], neuronsType)

    def test_read_write_weights(self):
        arch = [2, 2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, SigmoidNeuron]
        network = Network(arch, neuronsType)

        weights_1_0 = [0.15, 0.2, 0.35]
        weights_1_1 = [0.25, 0.3, 0.35]
        weights_2_0 = [0.4, 0.45, 0.6]
        weights_2_1 = [0.5, 0.55, 0.6]

        # set weights
        network.layers[1].neurons[0].weights = weights_1_0
        network.layers[1].neurons[1].weights = weights_1_1
        network.layers[2].neurons[0].weights = weights_2_0
        network.layers[2].neurons[1].weights = weights_2_1

        # dump and re-read weights
        with open("test_weights.csv", "w") as out_file:
            network.dump_weights(out_file)
        with open("test_weights.csv", "r") as in_file:
            network.load_weights(in_file)

        # check that the weights are the same
        self.assertEqual(network.layers[1].neurons[0].weights, weights_1_0)
        self.assertEqual(network.layers[1].neurons[1].weights, weights_1_1)
        self.assertEqual(network.layers[2].neurons[0].weights, weights_2_0)
        self.assertEqual(network.layers[2].neurons[1].weights, weights_2_1)


if __name__ == '__main__':
    unittest.main()
