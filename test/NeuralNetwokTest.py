import unittest
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from Neural_network import *

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
        network.update_weights_SGD(delta_w, learning_rate=0.5/2, prev_delta=np.zeros(delta_w.shape), momentum=0)

        layers = network.layers
        self.assert_weights(layers)
        self.assertEqual(loss_value.round(9) / 2, 0.298371109)  # divide the obtained loss by two

    def test_train_SGD(self):
        arch = [2, 2, 2]
        neuronsType = [InputNeuron, SigmoidNeuron, SigmoidNeuron]
        network = Network(arch, neuronsType)
        self.set_weights(network)

        data = [[0.05, 0.1]]
        target = np.array([[0.01, 0.99]])
        tr_l, _, _, _, _ = network.train_SGD(x_train=data, y_train=target,
                                          x_test=data, y_test=target, lossObject=SquaredError("sigmoid"),
                                      epochs=1, learning_rate=0.5/2, batch_size=1, momentum=0, regularization=0, epsilon=1e-10)

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
        network = Network([1, 1], [InputNeuron, OutputNeuron])
        network.layers[1].neurons[0].weights = [2, 1]
        self.assertEqual(11, network.forward([5])[0])
        self.assertEqual(7, network.forward([3])[0])

    def test_little_training(self):
        network = Network([2,1], [InputNeuron, OutputNeuron])
        network.layers[1].neurons[0].weights = np.array([0.5,0.2,0.3])
        data = [[2,2]]
        target = np.array([[4]])
        self.assertEqual(network.predict(data)[0][0], 1.7)
        loss, _, _, _, _ = network.train_SGD(data, target, data, target, SquaredError("sigmoid"), 1, 0.01, 1, 0, 0, epsilon=1e-10)
        self.assertEqual(np.round(loss[0], 2), 5.29)
        self.assertEqual(np.round(network.predict(data)[0][0], 3), 2.114)

    def test_little_bfgs_training(self):
        # AAA assumes that the interpolation takes the middle value between alpha_low and alpha_high
        network = Network([2, 1], [InputNeuron, OutputNeuron])
        network.layers[1].neurons[0].weights = np.array([0.5, 0.2, 0.3])
        data = [[2, 2]]
        target = np.array([[4]])
        loss, _, _, _ , _, _, _ = network.train_BFGS(data, target, data, target, theta=0.9, c_1=0.0001, c_2=0.9,
                                          lossObject=SquaredError("sigmoid"), epochs=1, regularization=0,
                                          epsilon=0.0001)
        self.assertEqual(np.round(loss[1], 8), 0.08265625)
        self.assertAlmostEqual(network.predict(data)[0][0], 4.2875)

    def test_bfgs_equal_lbfgs_if_m_is_big(self):
        x_train = [[2, 2], [1, 1], [3, 2], [2, 4]]
        y_train = np.array([[4], [2], [5], [6]])
        arch = [2, 1]
        neurons = [InputNeuron, OutputNeuron]
        lossObject=SquaredError("sigmoid")

        network_bfgs = Network(arch, neurons)
        network_lbfgs = copy.deepcopy(network_bfgs)

        loss_bfgs, _, _, _, _, _, _ = network_bfgs.train_BFGS(x_train, y_train, x_train, y_train, theta=0.9, c_1=0.0001,
                                                    c_2=0.9, epsilon=1e-10, lossObject=lossObject,
                                                    epochs=5, regularization=0)
        loss_lbfgs, _, _, _, _, _, _ = network_lbfgs.train_LBFGS(x_train, y_train, x_train, y_train, lossObject=lossObject,
                                                       m=4, epochs=5, regularization=0, theta=0.9,
                                                       c_1=0.0001, c_2=0.9, epsilon=1e-10, is_test=True)
        # this two should be the same for m greater than epochs
        np.testing.assert_almost_equal(loss_bfgs, loss_lbfgs, decimal=5)

        # equality of gradients
        gradient_bfgs, _, _ = network_bfgs.calculate_gradient(x_train, y_train, lossObject, regularization=0)
        gradient_lbfgs, _, _ = network_lbfgs.calculate_gradient(x_train, y_train, lossObject, regularization=0)

        np.testing.assert_almost_equal(gradient_bfgs, gradient_lbfgs, decimal=5)

    def test_gradient_1_layer_linear(self):
        network = Network([2,2], [InputNeuron, OutputNeuron])
        network.layers[-1].neurons[0].weights = np.array([3, 2, 1], dtype=np.float)
        network.layers[-1].neurons[1].weights = np.array([4, 1, 1], dtype=np.float)

        x, y = [[1,2]], np.array([[3,3]])
        loss_obj = SquaredError()
        # computation sub-routines
        output_net = network.predict(x)
        loss_derivative = loss_obj.derivative(y, output_net)[0]
        delta_out_layer = network.compute_delta_output_layer(output_net, y, loss_obj)[0]
        gradient, _, _ = network.back_propagation(y[0], lossObject=loss_obj, regularization=0)
        gradient_vector, _, _ = network.calculate_gradient(x, y, loss_obj, regularization=0)
        # assertion
        np.testing.assert_array_equal(output_net[0], [8,7])
        np.testing.assert_array_equal(loss_derivative, [10, 8])
        np.testing.assert_array_equal(delta_out_layer, [10, 8])
        np.testing.assert_array_equal(gradient.tolist(), [[[10, 20, 10], [8, 16, 8]]])
        np.testing.assert_array_equal(gradient_vector, [10, 20, 10, 8, 16, 8])

    def test_gradient_2_layer_linear(self):
        network = Network([2,2,2], [InputNeuron, LinearNeuron, OutputNeuron])
        network.layers[-2].neurons[0].weights = np.array([3, 2, 1], dtype=np.float)
        network.layers[-2].neurons[1].weights = np.array([4, 1, 1], dtype=np.float)
        network.layers[-1].neurons[0].weights = np.array([1, 0, 1], dtype=np.float)
        network.layers[-1].neurons[1].weights = np.array([2, 1, 1], dtype=np.float)

        x, y = [[1,2]], np.array([[10,26]])
        loss_obj = SquaredError()
        # computation sub-routines
        output_net = network.predict(x)
        loss_derivative = loss_obj.derivative(y, output_net)[0]
        delta_out_layer = network.compute_delta_output_layer(output_net, y, loss_obj)[0]
        gradient, _, _ = network.back_propagation(y[0], lossObject=loss_obj, regularization=0)
        gradient_vector, _, _ = network.calculate_gradient(x, y, loss_obj, regularization=0)
        # assertion
        np.testing.assert_array_equal(output_net[0], [9,24])
        np.testing.assert_array_equal(loss_derivative, [-2, -4])
        np.testing.assert_array_equal(delta_out_layer, [-2, -4])
        np.testing.assert_array_equal(gradient.tolist(), [[[-10, -20, -10], [-4, -8, -4]], [[-16,-14,-2], [-32,-28,-4]]])
        np.testing.assert_array_equal(gradient_vector, [-10, -20, -10,-4, -8, -4,-16,-14,-2, -32,-28,-4])

    def test_init_zero_gradient(self):
        network = Network([2,5,1], [InputNeuron, TanHNeuron, OutputNeuron])
        g_init = network.zero_init_gradient()
        np.testing.assert_array_equal((g_init[0].shape, g_init[1].shape), ((5,3), (1,6)))

        network = Network([2,1,1], [InputNeuron, TanHNeuron, OutputNeuron])
        g_init = network.zero_init_gradient()
        np.testing.assert_array_equal((g_init[0].shape, g_init[1].shape), ((1,3), (1,2)))

    def test_gradient_2_layer_tanh_lin(self):
        network = Network([2,1,1], [InputNeuron, TanHNeuron, OutputNeuron])
        network.layers[-2].neurons[0].weights = np.array([1, 2, 1], dtype=np.float)
        network.layers[-1].neurons[0].weights = np.array([2, 1], dtype=np.float)

        x, y = [[2,1]], np.array([[4]])
        loss_obj = SquaredError()

        # computation sub-routines
        output_net = network.predict(x)
        loss_derivative = loss_obj.derivative(y, output_net)[0]
        delta_out_layer = network.compute_delta_output_layer(output_net, y, loss_obj)[0]
        delta_hid_layer = network.compute_delta_hidden_layer(delta_out_layer, currentLayerIndex=1)
        gradient_vector, _, _ = network.calculate_gradient(x, y, loss_obj, regularization=0)
        #gradient, _, _ = network.back_propagation(y[0], lossObject=EuclideanError(), regularization=0)

        # assertion
        np.testing.assert_array_almost_equal(output_net[0], [2.9998184])
        np.testing.assert_array_almost_equal(loss_derivative, [-2.0003632])
        np.testing.assert_array_almost_equal(delta_out_layer, [-2.0003632])
        np.testing.assert_array_almost_equal(delta_hid_layer, [-0.0007264648])
        np.testing.assert_array_almost_equal(gradient_vector, [-0.00145293, -0.00072646, -0.00072646, -2.00018158, -2.0003632])
        #gradient_paper = np.reshape([[[-0.00145293, -0.00072646, -0.00072646]], [[-2.00018158, -2.0003632]]], newshape=(2,))
        #np.testing.assert_array_almost_equal(gradient, gradient_paper)

    def test_two_layers_2_patterns(self):
        network = Network([2,2,1], [InputNeuron, LinearNeuron, OutputNeuron])
        network.layers[1].neurons[0].weights = np.array([1, 3, 1], dtype=np.float)
        network.layers[1].neurons[1].weights = np.array([1, 0, 1], dtype=np.float)
        network.layers[2].neurons[0].weights = np.array([1, 1, 1], dtype=np.float)

        x, y = [[1,1], [2,2]], np.array([[2],[3]])
        loss_obj = EuclideanError()

        output_net = network.predict(x)
        loss_value = loss_obj.value(y, output_net)
        gradient_vector, _, _ = network.calculate_gradient(x, y, loss_obj, regularization=0)

        np.testing.assert_array_almost_equal(output_net, [[8], [13]])
        self.assertAlmostEqual(loss_value, 11.661903789)
        np.testing.assert_array_almost_equal(gradient_vector, [1.5, 1.5, 1, 1.5, 1.5, 1, 7, 2.5, 1])

    def test_evaluate_phi_alpha(self):
        network = Network([2,2], [InputNeuron, OutputNeuron])
        network.layers[-1].neurons[0].weights = np.array([3, 2, 1], dtype=np.float)
        network.layers[-1].neurons[1].weights = np.array([4, 1, 1], dtype=np.float)

        x, y = [[1,2]], np.array([[3,3]])
        loss_obj = EuclideanError()
        gradient_vector, _, _ = network.calculate_gradient(x, y, loss_obj, regularization=0)
        H = np.identity(gradient_vector.shape[0])
        p = - H.dot(gradient_vector)

        layers = copy.deepcopy(network.get_weights_as_vector())
        g_phi_alpha, phi_alpha = network.phi_alpha(0.5, x, loss_obj, p, y, regularization=0)
        _, phi_0 = network.phi_alpha(0.0, x, loss_obj, p, y, regularization=0)
        loss = loss_obj.value(y, network.predict(x))

        np.testing.assert_array_almost_equal(layers, network.get_weights_as_vector())
        self.assertAlmostEqual(phi_alpha, 3.4031242374328494) #32.0156212)
        np.testing.assert_array_almost_equal(g_phi_alpha, [0.78086881, 1.56173762, 0.78086881, 0.62469505, 1.2493901 , 0.62469505])#np.array([-50,-100,-50,-40,-80,-40]))
        self.assertAlmostEqual(phi_0, loss)

    def update_matrix_BFGS(self):
        dummy_network = Network([1,1], [InputNeuron, OutputNeuron])

        H_k = np.array([[4,2], [2,3]])
        s_k = np.array([1,2])
        y_k = np.array([3,1])

        # test H pos def and simmetric
        self.assertEqual(np.all(np.linalg.eigvals(H_k) > 0), True)
        np.testing.array_almost_equal(H_k, H_k.T)

        # update matrix
        H_new = dummy_network.update_matrix_BFGS(H_k, s_k, y_k)

        # test H_new simmetric, pos def, secant equation
        np.testing.array_almost_equal(H_new, np.array([[11, -33], [-33, 99]]) / 25)
        self.assertEqual(np.all(np.linalg.eigvals(H_new) > 0), True)
        np.testing.array_almost_equal(H_new, H_new.T)
        np.testing.assert_array_almost_equal(np.dot(H_new, y_k), s_k)

    def test_regularization(self):
        network = Network([2,1], [InputNeuron, OutputNeuron])
        network.layers[1].neurons[0].weights = np.array([3, 2, 1], dtype=np.float)

        x, y = [[1,2], [2,2]], np.array([[3],[4]])
        gradient_mse, loss_value_mse, _ = network.calculate_gradient(x, y, SquaredError(), regularization=0.01)
        gradient_mse_no_reg, loss_value_mse_no_reg, _ = network.calculate_gradient(x, y, SquaredError(), regularization=0.0)
        gradient_mee, loss_value_mee, _ = network.calculate_gradient(x, y, EuclideanError(), regularization=0.01)
        gradient_mee_no_reg, loss_value_mee_no_reg, _ = network.calculate_gradient(x, y, EuclideanError(), regularization=0.0)

        self.assertAlmostEqual(loss_value_mse, 37.13)
        self.assertAlmostEqual(loss_value_mse_no_reg, 37)
        self.assertAlmostEqual(loss_value_mee, 6.13)
        self.assertAlmostEqual(loss_value_mee_no_reg, 6)
        np.testing.assert_array_almost_equal(gradient_mse, np.array([19.06, 24.04, 12]))
        np.testing.assert_array_almost_equal(gradient_mse_no_reg, np.array([19, 24, 12]))
        np.testing.assert_array_almost_equal(gradient_mee, np.array([1.56, 2.04, 1]))
        np.testing.assert_array_almost_equal(gradient_mee_no_reg, np.array([1.5, 2, 1]))


if __name__ == '__main__':
    unittest.main()
