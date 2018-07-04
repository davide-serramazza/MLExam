import unittest
import matplotlib.pyplot as plt
from src.Neural_network import *

class TestNeuralNetwork(unittest.TestCase):
    def test_forward(self):
        arch = [2, 2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, SigmoidNeuron]
        data = [0.05, 0.1]
        network = Network(arch, neuronsType)

        self.set_weights(network)

        network.forward(data)
        network_output = (network.getOutput())
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
        delta_w, loss_value, _ = network.back_propagation(target=target, lossObject=SquaredError("sigmoid"), regularization=0)
        network.update_weights(delta_w, learning_rate=0.5/2, prev_delta=np.zeros(delta_w.shape), momentum=0)

        layers = network.layers
        self.assert_weights(layers)
        self.assertEqual(loss_value.round(9) / 2, 0.298371109)  # divide the obtained loss by two


    def test_train(self):
        arch = [2, 2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, SigmoidNeuron]
        network = Network(arch, neuronsType)
        self.set_weights(network)

        data = [[0.05, 0.1]]
        target = [[0.01, 0.99]]
        tr_l, _, _, _ = network.train(data=data, targets=target, eval_data=[], eval_targets=[], lossObject=SquaredError("sigmoid"),
                      epochs=1, learning_rate=0.5/2, batch_size=1, momentum=0, regularization=0)

        self.assert_weights(network.layers)

    def assert_weights(self, layers):
        np.testing.assert_array_equal(layers[1].neurons[0].weights, [0.14978071613276281, 0.19956143226552567, 0.34561432265525649])
        np.testing.assert_array_equal(layers[1].neurons[1].weights, [0.24975114363236958, 0.29950228726473915, 0.3450228726473914])
        np.testing.assert_array_equal(layers[2].neurons[0].weights, [0.35891647971788465, 0.4086661860762334, 0.53075071918572148])
        np.testing.assert_array_equal(layers[2].neurons[1].weights, [0.5113012702387375, 0.56137012110798912, 0.61904911825827813])

    def set_weights(self, network):
        network.layers[1].neurons[0].weights = np.asarray([0.15, 0.2, 0.35])
        network.layers[1].neurons[1].weights = np.asarray([0.25, 0.3, 0.35])
        network.layers[2].neurons[0].weights = np.asarray([0.4, 0.45, 0.6])
        network.layers[2].neurons[1].weights = np.asarray([0.5, 0.55, 0.6])

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

        weights_1_0 = np.array([0.15, 0.2, 0.35])
        weights_1_1 = np.array([0.25, 0.3, 0.35])
        weights_2_0 = np.array([0.4, 0.45, 0.6])
        weights_2_1 = np.array([0.5, 0.55, 0.6])

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
        np.testing.assert_array_equal(network.layers[1].neurons[0].weights, weights_1_0)
        np.testing.assert_array_equal(network.layers[1].neurons[1].weights, weights_1_1)
        np.testing.assert_array_equal(network.layers[2].neurons[0].weights, weights_2_0)
        np.testing.assert_array_equal(network.layers[2].neurons[1].weights, weights_2_1)

    def test_dummy_forward(self):
        network = Network([1,1], [InputNeuron, OutputNeuron])
        network.layers[1].neurons[0].weights = [2, 1]
        self.assertEqual(11, network.forward([5])[0])
        self.assertEqual(7, network.forward([3])[0])

    def test_little_training(self):
        network = Network([2,1], [InputNeuron, OutputNeuron])
        network.layers[1].neurons[0].weights = np.array([0.5,0.2,0.3])
        data = [[2,2]]
        target = [[4]]
        self.assertEqual(network.predict(data)[0][0], 1.7)
        loss, _, _, _ = network.train(data, target, [], [], SquaredError("sigmoid"), 1, 0.01, 1, 0, 0)
        self.assertEqual(np.round(loss[0], 2), 5.29)
        self.assertEqual(np.round(network.predict(data)[0][0], 3), 2.114)

    def test_little_bfgs_training(self):
        # AAA assumes that the interpolation takes the middle value between alpha_low and alpha_high
        network = Network([2, 1], [InputNeuron, OutputNeuron])
        network.layers[1].neurons[0].weights = np.array([0.5, 0.2, 0.3])
        data = [[2,2]]
        target = [[4]]
        loss, _, _, _ = network.trainBFGS(data, target, [], [], 0.9, 0.0001, 0.9, SquaredError("sigmoid"), 1, 0)
        self.assertEqual(np.round(loss[1], 8), 0.08265625)
        self.assertEqual(network.predict(data)[0][0], 4.2875)

    def test_bfgs_equal_lbfgs_if_m_is_big(self):
        data = [[2, 2], [1, 1], [3, 2], [2, 4]]
        target = [[4], [2], [5], [6]]
        arch = [2, 1]
        neurons = [InputNeuron, OutputNeuron]

        network_bfgs = Network(arch, neurons)
        network_lbfgs = Network(arch, neurons)

        network_bfgs.layers[1].neurons[0].weights = np.array([0.5, 0.2, 0.3])
        network_lbfgs.layers[1].neurons[0].weights = np.array([0.5, 0.2, 0.3])

        loss_bfgs, _, _, _ = network_bfgs.trainBFGS(data, target, [], [], theta=0.9, c_1=0.0001, c_2=0.9,
                                                    lossObject=SquaredError("sigmoid"), epochs=5, regularization=0)
        loss_lbfgs, _, _, _ = network_lbfgs.trainLBFGS(data, target, [], [], SquaredError("sigmoid"),
                                                       m=10, epochs=5, regularization=0, theta=0.9,
                                                       c_1=0.0001, c_2=0.9, alpha_0=1)
        # TODO this two should be the same
        print loss_bfgs
        print loss_lbfgs




if __name__ == '__main__':
    unittest.main()
