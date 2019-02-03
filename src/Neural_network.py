from Layer import *
from loss_functions import *
import sys
from collections import deque
from numpy.linalg import cond, norm
import matplotlib.pyplot as plt
from utils import shuffle_dataset, is_pos_def


class Network:
    def __init__(self, architecture, neurons):
        """
        Initialize neural network.
        :argument
        architecture - list encoding the architecture. The i-th entry is the number of neurons at the i-th layer.
        neurons -  list of neurons type. The i-th entry is the type of neurons of the i-th layer.
        """
        check_topology(architecture, neurons)
        self.layers = []
        self.architecture = architecture  # for quick access when writing weights to file

        # input layer
        inputNeuron = neurons[0](0)     # Input neurons don't have weights
        layer = Layer(architecture[0], 0, inputNeuron)
        self.layers.append(layer)
        # hidden and output layers
        for i in range(1, len(architecture)):
            len_weights = len(self.layers[i-1].neurons)
            neuron = neurons[i](len_weights=len_weights)
            layer = Layer(architecture[i], architecture[i - 1], neuron)
            self.layers.append(layer)

    def getOutput(self):
        """
        Returns the computed scores of the neural network
        (i.e the output of the neurons of the last layer).

        :return: outputs of the neural network
        """
        last_layer = self.layers[-1]
        return last_layer.getOutput()[:-1]

    def forward(self, pattern):
        """
        Computes output of the neural network as a function of the data pattern.
        :param pattern: a single training example
        :return: output of the neural network
        """
        self.feed_input_neurons(pattern)
        self.propagate_input()
        return self.getOutput()

    def propagate_input(self):
        """
        propagate input through the network
        :return:
        """
        # propagate result
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for neuron in layer.neurons[:-1]:  # exclude bias
                weights = neuron.weights
                input_x = self.layers[i - 1].getOutput()
                scores = np.dot(weights, input_x)
                neuron.activation_function(scores)

    def feed_input_neurons(self, pattern):
        """
        Set the output of the input neurons to the current pattern.
        Activation function for input neurons is the identity.
        :param pattern: single training example
        :return:
        """
        input_layer = self.layers[0]
        for input_neuron, x in zip(input_layer.neurons[:-1], pattern):  # exclude bias
            input_neuron.activation_function(x)

    def back_propagation(self, target, lossObject, regularization):
        """
        Performs backpropagation.
        :param target: target vector for a single training example
        :param lossObject: function to optimize
        :param regularization: regularization strength
        :return: gradient_weights: gradient w.r.t network weights
                 loss_value: loss value computed by lossObject
                 misClassification: misclassification error
        """
        # vector containing the deltas of each layer
        delta_vectors = deque()

        # 1. get output vector from forward step
        output_net = np.array(self.getOutput())

        # 2. for each network output unit compute its error term delta
        delta_outputLayer = self.compute_delta_output_layer(output_net, target, lossObject)
        delta_vectors.append(delta_outputLayer)

        # 3. for each hidden unit compute its error term delta
        delta_next_layer = delta_outputLayer
        # compute delta vector for each hidden layer and append it to delta_vectors
        for hidden_layer_index in range(len(self.layers) - 2, 0, -1):
            delta_next_layer = self.compute_delta_hidden_layer(delta_next_layer, hidden_layer_index)
            delta_vectors.appendleft(delta_next_layer)

        # 4. compute network weights update
        gradient_weights = self.compute_gradient_from_deltas(np.asarray(delta_vectors))

        # add regularization gradient 2 * lambda * w_{ji}
        for grad_layer, layer in zip(gradient_weights, self.layers[1:]):  # exclude input layer
            for grad_neuron, neuron in zip(grad_layer, layer.neurons[:-1]):  # exclude bias neuron
                reg_component_vector = 2 * regularization * neuron.weights[:-1]  # exclude bias neuron weight
                grad_neuron[:-1] = grad_neuron[:-1] + reg_component_vector

        # 5 report loss and misclassification count
        weights = self.get_weights_as_vector()
        loss_value = lossObject.value(target, output_net, weights, regularization)
        misClassification = lossObject.misClassification(target, output_net)

        return gradient_weights, loss_value, misClassification

    def get_weights_as_vector(self):
        """
        gets neural network weights as a single array.
        :return: array of weights
        """
        weights = [neuron.weights for layer in self.layers[1:] for neuron in layer.neurons[:-1]]
        weights = np.concatenate(weights).ravel()
        return weights

    def compute_gradient_from_deltas(self, delta_vectors):
        """
        Computes the gradient from the delta of each neuron.
        :param delta_vectors: vector where each entry delta_vectors[l][n]
                contains the delta of each neuron 'n' in layer 'l'.
        :return: a matrix gradient_w where each entry gradient[l][n][w] is the gradient w.r.t
                the weight 'w' of the neuron 'n' in the layer 'l'.
        """
        gradient_w = np.empty((len(delta_vectors)), dtype=object)

        for l in range(1, len(self.layers)):
            tmpL = np.empty((len(self.layers[l].neurons[:-1]),len(self.layers[l-1].neurons)), dtype=float)
            for n in range(len(self.layers[l].neurons[:-1])):  # exclude bias neuron
                neuron_input = np.array([neuron.getOutput() for neuron in self.layers[l-1].neurons], dtype=float)
                tmpL[n,:] = neuron_input * delta_vectors[l-1][n]
            gradient_w[l-1] = tmpL
        return gradient_w

    def update_weights_SGD(self, gradient_w, learning_rate, prev_delta, momentum):
        """
        update weights as
            DeltaW_{ji} = - learning_rate * gradient_w + momentum * prev_delta
            w_{ji} += DeltaW_{ji}
        :param gradient_w: gradient error on data wrt neural network`s weights
        :param learning_rate: learning rate
        :param prev_delta: previous weight update, for momentum
        :param momentum: percentage of prev_delta
        :return: delta_w, current weight update (i.e prev_delta for next iteration)
        """
        delta_w = - learning_rate * gradient_w + momentum * prev_delta
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons) - 1):
                self.layers[i].neurons[j].weights = self.layers[i].neurons[j].weights + delta_w[i-1][j]
        return delta_w

    def compute_delta_hidden_layer(self, delta_next_layer, currentLayerIndex):
        # delta_layer vector
        delta_layer = np.empty(shape=(len(self.layers[currentLayerIndex].neurons)-1))
        for h in range(len(self.layers[currentLayerIndex].neurons)-1):
            downstream = self.layers[currentLayerIndex + 1].neurons[:-1]
            weights = [neuron.weights[h] for neuron in downstream]
            gradient_flow = np.dot(weights, delta_next_layer)
            derivative_net = self.layers[currentLayerIndex].neurons[h].activation_function_derivative()
            delta_h = gradient_flow * derivative_net
            delta_layer[h] = delta_h
        return delta_layer

    def compute_delta_output_layer(self, output_net, target, loss):
        output_layer = self.layers[-1]
        af_derivatives = np.array([neuron.activation_function_derivative() for neuron in output_layer.neurons[:-1]])
        error_derivatives = loss.derivative(np.array(target), output_net)
        delta_outputLayer = np.multiply(af_derivatives, error_derivatives)
        return delta_outputLayer

    def validation_error(self, patterns, targets, loss_obj):
        """
        compute validation loss and misclassification error
        :param patterns: validation set patterns
        :param targets: validation set targets
        :param loss_obj: loss for computing error
        :return: validation loss
                 validation misclassification error
        """
        # predict each validation set pattern
        scores = self.predict(patterns)

        loss_epoch, missclass_epoch = 0, 0
        # add errors
        for i in range(len(scores)):
            loss_epoch += loss_obj.value(targets[i], scores[i])
            missclass_epoch += loss_obj.misClassification(targets[i], scores[i])

        # average
        loss_epoch /= float(len(scores))
        missclass_epoch /= float(len(scores))

        return loss_epoch, missclass_epoch

    def train_SGD(self, x_train, y_train, x_test, y_test, lossObject, epochs, learning_rate,
                  batch_size, momentum, regularization, epsilon):
        """
        Performs the training of the neural network.
        :param x_train: traning patterns
        :param y_train: traning targets
        :param x_test : test patterns
        :param y_test:  test targets
        :param lossObject: loss
        :param epochs: iterations to train
        :param learning_rate: step size
        :param batch_size: size of the batch to forward before updating the weights
        :param momentum: percentage of previous weight update to add
        :param regularization: regularization strength
        :return: losses, vector of the loss computed at each epoch
                 misClassification, vector of misclassification loss for each epoch
        """

        # lists for specify missclassification and Squared error (for training and validation)
        losses, losses_val = np.array([]), np.array([])
        missclass, missclass_val = np.array([]), np.array([])
        gradient_norm = []

        # prev_delta is previous weights update (for momentum)
        prev_delta = self.zero_init_gradient()

        for epoch in range(epochs):
            # current epoch value of misclassification and Squared error
            loss_epoch, missclass_epoch, gradient_norm_epoch = 0, 0, 0

            # shuffle data set
            data_shuffled, targets_shuffled = shuffle_dataset(x_train, y_train)

            for i in range(0, len(data_shuffled), batch_size):
                # take only batch_size examples
                batch_pattern = data_shuffled[i:i + batch_size]
                batch_target = targets_shuffled[i:i + batch_size]

                # gradient_w_batch = sum of gradient_w for the epoch
                gradient_w_batch = self.zero_init_gradient()

                # train, compute gradient for a batch
                for pattern, t in zip(batch_pattern, batch_target):
                    self.forward(pattern)
                    gradient_w, loss_p, miss_p = self.back_propagation(t, lossObject, regularization)
                    loss_epoch += loss_p
                    missclass_epoch += miss_p
                    gradient_w_batch += gradient_w

                gradient_w_batch /= float(len(batch_pattern))  # take mean gradient across batch
                gradient_norm_epoch += np.linalg.norm(self.get_gradient_as_vector(gradient_w_batch))

                # update neural network weights after a batch of training example
                # save previous weight update
                prev_delta = self.update_weights_SGD(gradient_w=gradient_w_batch,
                                                 learning_rate=learning_rate,
                                                 prev_delta=prev_delta,
                                                 momentum=momentum)

            # append the total loss and misClassification of single epoch
            losses = np.append(losses, loss_epoch)
            missclass = np.append(missclass, missclass_epoch)
            gradient_norm.append(gradient_norm_epoch)

            # computing loss and misClassification on validation set then append to list
            loss_val_epoch, missclass_val_epoch = self.validation_error(x_test, y_test, lossObject)

            losses_val = np.append(losses_val, loss_val_epoch)
            missclass_val = np.append(missclass_val, missclass_val_epoch)

            # condition stop
            if gradient_norm_epoch < epsilon:
                print "stop: norm gradient. Epoch", epoch
                break

        # end of training - average over training set
        losses /= float(len(x_train))
        missclass /= float(len(x_train))

        return losses, missclass, losses_val, missclass_val, gradient_norm

    # --------------------- BFGS & L-BFGS ---------------------------------- #

    def get_gradient_as_vector(self, list_of_lists):
        """
        flattens a list of lists into a one-dimensional array
        :param list_of_lists:
        :return:
        """
        gradient_vector = [n for l in list_of_lists for n in l]
        gradient_vector = np.concatenate(gradient_vector).ravel()
        return gradient_vector

    def update_weights_BFGS(self, delta):
        """
        update network weights (used by BFGS and L-BFGS)
        :param delta: weight update (alpha_k * p_k)
        :return: x_k+1
        """
        start = 0

        for layer in self.layers[1:]:  # exclude input layer
            for neuron in layer.neurons[:-1]:  # exclude bias neuron
                current_neuron_weights = neuron.weights

                # taking only gradient's entry w.r.t. current gradient
                weights_len = len(current_neuron_weights)
                tmp = delta[start:start + weights_len]
                start += weights_len
                # update weights
                current_neuron_weights += tmp

        return self.get_weights_as_vector()

    def calculate_gradient(self, data, targets, lossObject, regularization):
        """
        computes gradient on entire training set
        :param data:
        :param targets:
        :param lossObject:
        :param regularization:
        :return: gradient,
                 loss,
                 misclassification
        """
        # create empty vector, gradient_w_old = sum of gradient_w for the epoch
        gradient_w_batch = self.zero_init_gradient()
        loss_batch, miss_batch = 0, 0

        for pattern, t in zip(data, targets):
            # calculate derivative for every pattern, then append to gradient_w_batch
            self.forward(pattern)
            gradient_w, loss_p, miss_p = self.back_propagation(t, lossObject, regularization)
            gradient_w_batch += gradient_w
            loss_batch += loss_p
            miss_batch += miss_p

        # getting the gradient as vector
        gradient = self.get_gradient_as_vector(gradient_w_batch)

        # compute mean values
        gradient /= float(len(data))
        loss_batch /= float(len(data))
        miss_batch /= float(len(data))

        return gradient, loss_batch, miss_batch

    def zero_init_gradient(self):
        """
        initialize a gradient-placeholder to zero
        :return:
        """
        a = self.architecture
        #zero_gradient = np.array([np.zeros((a[i], a[i - 1] + 1)) for i in range(1, len(a))])
        zero_gradient = np.empty(len(a)-1, dtype=object)
        for i in range(1, len(a)):
            zero_gradient[i-1] = np.zeros((a[i], a[i - 1] + 1))
        return zero_gradient

    def update_matrix_BFGS(self, H_k, s_k, y_k):
        """
        BFGS matrix update.
        :param H_k: current inverse of the approximation of the true Hessian
        :param s_k:
        :param y_k:
        :return: new inverse of the approximation of the Hessian
        """
        # get dimension
        shape = H_k.shape[0]

        # rho_k = 1/(y_k^t*s_k)
        rho_k = float(1) / np.dot(s_k, y_k)

        if rho_k < 0:
            raise Exception("rho_K < 0")

        # V_k = I - rho_k * s_k * y_k^t
        tmp = rho_k * np.outer(y_k, s_k)
        V_k = np.identity(shape) - tmp

        # H_{k+1} = V_k^t * H_k * V_k + rho_k * s_k * s_k^t
        tmp = np.dot(V_k.T, H_k)
        H_k = np.dot(tmp, V_k)
        H_k = H_k + rho_k * np.outer(s_k, s_k)

        return H_k

    def train_BFGS(self, x_train, y_train, x_test, y_test, theta, c_1, c_2,
                  lossObject, epochs, regularization, epsilon, line_search='wolfe', debug=False):
        # allocate arrays
        alpha_list = np.empty((epochs+1), dtype=object) # list that holds the step lengths alpha_k taken at each epoch
        norm_gradient = np.empty((epochs+1), dtype=object)
        condition_numbers = np.empty((epochs+1), dtype=object)
        tr_losses = np.empty((epochs+1), dtype=object)
        vl_losses = np.empty((epochs+1), dtype=object)
        tr_misses = np.empty((epochs+1), dtype=object)
        vl_misses = np.empty((epochs+1), dtype=object)

        # 1. compute initial gradient and initial Hessian approximation H_0
        gradient_old, loss, miss = self.calculate_gradient(x_train, y_train, lossObject, regularization)
        H = np.identity(gradient_old.shape[0])
        x_old = self.get_weights_as_vector()

        # append train / validation losses
        loss_val_epoch, misclass_val_epoch = self.validation_error(x_test, y_test, lossObject)
        tr_losses[0] = loss
        tr_misses[0] = miss
        vl_losses[0] = loss_val_epoch
        vl_misses[0] = misclass_val_epoch
        norm_gradient[0] = norm(gradient_old)
        condition_numbers[0] = cond(H)

        for epoch in range(epochs):
            print epoch, "out of", epochs

            # stop criterion
            if (norm(gradient_old)) < epsilon:
                print "stop: gradient norm, epoch", epoch
                break

            # 1. compute search direction p = -H * gradient
            p = - H.dot(gradient_old)
            #TODO: try normalize p
            #p /= norm(p)
            #print - np.dot(gradient_old, p) / (norm(gradient_old) * norm(p)) >= 1 / cond(H)

            # 2. line search
            if line_search == 'wolfe':
                alpha = self.armijo_wolfe_line_search(c_1, c_2, x_train, gradient_old, loss, lossObject, p, y_train, theta, regularization)
            elif line_search == 'backtracking':
                alpha = self.backtracking_line_search(c_1, x_train, gradient_old, loss, lossObject, p, y_train, theta, regularization)

            if alpha == -1:
                print "stop: line search, epoch", epoch
                if debug:
                    self.plot_phi_alpha_in_neighborhood(x_train, lossObject, p, y_train, regularization)
                    self.plot_phi_alpha_and_tangent_line(x_train, lossObject, y_train, p, gradient_old, regularization)
                break
            alpha_list[epoch] = alpha

            # 3. update weights using x_{k+1} = x_{k} + alpha_{k} * p_k
            x_new = self.update_weights_BFGS(delta=alpha * p)

            # 4. compute new gradient
            gradient_new, loss, miss = self.calculate_gradient(x_train, y_train, lossObject, regularization)

            # append training / validation losses
            loss_val_epoch, misclass_val_epoch = self.validation_error(x_test, y_test, lossObject)
            tr_losses[epoch+1] = loss
            tr_misses[epoch+1] = miss
            vl_losses[epoch+1] = loss_val_epoch
            vl_misses[epoch+1] = misclass_val_epoch
            norm_gradient[epoch+1] = norm(gradient_new)

            # 5. compute
            # s_k = x_{k+1} - x_k = x_new - x_old
            # y_k = nabla f_{k+1} - nabla f_k = gradient new - gradient old
            s_k = x_new - x_old
            y_k = gradient_new - gradient_old

            # 6. update matrix H
            H = self.update_matrix_BFGS(H, s_k, y_k)
            condition_numbers[epoch+1] = cond(H)
            if not is_pos_def(H):
                raise Exception("matrix H is not positive semidefinite")

            #np.testing.assert_array_almost_equal(np.dot(H,y_k), s_k)

            # update x_old and gradient_old
            x_old = x_new
            gradient_old = gradient_new

        # truncate arrays to proper length
        none = np.array(None)
        tr_losses = tr_losses[tr_losses != none]
        tr_misses = tr_misses[tr_misses != none]
        vl_losses = vl_losses[vl_losses != none]
        vl_misses = vl_misses[vl_misses != none]
        alpha_list = alpha_list[alpha_list != none]
        norm_gradient = norm_gradient[norm_gradient != none]
        condition_numbers = condition_numbers[condition_numbers != none]

        return tr_losses, tr_misses, vl_losses, vl_misses, alpha_list, norm_gradient, condition_numbers

    def compute_direction(self, H, gradient, s_list, y_list, rho_list):
        """
        computes direction with two-loops recursion (used by L-BFGS)
        :param H: current inverse of Hessian approximation
        :param gradient: gradient
        :param s_list:
        :param y_list:
        :param rho_list:
        :return: returns ascent direction
        """
        a_list = [0] * len(s_list)
        q = gradient
        # for i = k-1, ..., k-m
        for i in range(len(s_list) - 1, -1, -1):
            a = rho_list[i] * np.dot(s_list[i], q)
            a_list[i] = a
            q = q - a * y_list[i]

        r = H.dot(q)

        # for i = k-m, ..., k-1
        for i in range(len(s_list)):
            beta = rho_list[i] * np.dot(y_list[i], r)
            r = r + s_list[i] * (a_list[i] - beta)

        return r

    def train_LBFGS(self, x_train, y_train, x_test, y_test, lossObject, theta, c_1, c_2,
                    epsilon, m, regularization, epochs, line_search='wolfe', debug=False, is_test=False):
        # allocate arrays
        alpha_list = np.empty((epochs+1), dtype=object) # list that holds the step lengths alpha_k taken at each epoch
        norm_gradient = np.empty((epochs+1), dtype=object)
        condition_numbers = np.empty((epochs+1), dtype=object)
        tr_losses = np.empty((epochs+1), dtype=object)
        vl_losses = np.empty((epochs+1), dtype=object)
        tr_misses = np.empty((epochs+1), dtype=object)
        vl_misses = np.empty((epochs+1), dtype=object)

        # 1. compute initial gradient
        gradient_old, loss, miss = self.calculate_gradient(x_train, y_train, lossObject, regularization)
        x_old = self.get_weights_as_vector()

        # append train / validation losses
        loss_val_epoch, misclass_val_epoch = self.validation_error(x_test, y_test, lossObject)
        tr_losses[0] = loss
        tr_misses[0] = miss
        vl_losses[0] = loss_val_epoch
        vl_misses[0] = misclass_val_epoch
        norm_gradient[0] = norm(gradient_old)

        # set of current s, y, rho lists
        s_list, y_list, rho_list = deque([]), deque([]), deque([])

        # main loop
        for epoch in range(epochs):
            print epoch, "out of", epochs

            # stop criterion
            if (norm(gradient_old)) < epsilon:
                print "stop: norm gradient, epoch", epoch
                break

            # calculate central matrix {H_k}^0
            if epoch == 0 or is_test:
                H = np.identity(gradient_old.shape[0])
            else:
                num = np.dot(s_list[-1], y_list[-1])
                den = np.dot(y_list[-1], y_list[-1])
                gamma = num/den
                H = gamma * np.identity(gradient_old.shape[0])

            # compute condition number
            condition_numbers[epoch] = cond(H)
            if not is_pos_def(H):
                raise Exception("matrix H is not positive semidefinite")

            # compute p = - H_k * \nabla f_k using two loop recursion
            p = - self.compute_direction(H, gradient_old, s_list, y_list, rho_list)
            #TODO: try normalize p
            #p /= norm(p)
            #print - np.dot(gradient_old, p) / (norm(gradient_old) * norm(p)) >= 1 / cond(H)

            # line search
            if line_search == 'wolfe':
                alpha = self.armijo_wolfe_line_search(c_1, c_2, x_train, gradient_old, loss, lossObject, p, y_train, theta, regularization)
            elif line_search == 'backtracking':
                alpha = self.backtracking_line_search(c_1, x_train, gradient_old, loss, lossObject, p, y_train, theta, regularization)

            if alpha == -1:
                print "stop: line search, epoch", epoch
                if debug:
                    self.plot_phi_alpha_in_neighborhood(x_train, lossObject, p, y_train, regularization)
                    self.plot_phi_alpha_and_tangent_line(x_train, lossObject, y_train, p, gradient_old, regularization)
                break
            alpha_list[epoch] = alpha

            # updating weights and compute x_k+1 = x_k + a_k*p_k
            x_new = self.update_weights_BFGS(delta=alpha * p)
            gradient_new, loss, miss = self.calculate_gradient(x_train, y_train, lossObject, regularization)

            # append training / validation losses
            loss_val_epoch, misclass_val_epoch = self.validation_error(x_test, y_test, lossObject)
            tr_losses[epoch+1] = loss
            tr_misses[epoch+1] = miss
            vl_losses[epoch+1] = loss_val_epoch
            vl_misses[epoch+1] = misclass_val_epoch
            norm_gradient[epoch+1] = norm(gradient_new)

            if epoch > (m-1):
                # discard first element
                s_list.popleft()
                y_list.popleft()
                rho_list.popleft()

            # compute s_k , y_k, p_k
            s_k = x_new - x_old
            y_k = gradient_new - gradient_old
            rho_k = 1.0 / np.dot(s_k, y_k)

            if rho_k < 0:
                raise Exception("rho_K < 0")

            # append to vector
            s_list.append(s_k)
            y_list.append(y_k)
            rho_list.append(rho_k)

            # update x_old and gradient_old
            x_old = x_new
            gradient_old = gradient_new

        # truncate arrays to proper length
        none = np.array(None)
        tr_losses = tr_losses[tr_losses != none]
        tr_misses = tr_misses[tr_misses != none]
        vl_losses = vl_losses[vl_losses != none]
        vl_misses = vl_misses[vl_misses != none]
        alpha_list = alpha_list[alpha_list != none]
        norm_gradient = norm_gradient[norm_gradient != none]
        condition_numbers = condition_numbers[condition_numbers != none]

        return tr_losses, tr_misses, vl_losses, vl_misses, alpha_list, norm_gradient, condition_numbers

    def backtracking_line_search(self, c_1, data, gradient, loss,
                                 lossObject, p, targets, theta, regularization):
        """
        Performs a backtracking line search, along the search direction p, that satisfies the Armijo
        condition, starting from the initial step alpha
        :param alpha: initial step
        :param c_1: sufficient decrease condition parameter
        :param data:
        :param gradient:
        :param loss:
        :param lossObject:
        :param p: descent direction
        :param targets:
        :param theta: alpha multiplier
        :param regularization:
        :return: returns a step length satisfying the Armijo condition or -1
        """
        assert theta < 1
        phi_0 = loss  # phi(0) = f(x_k + 0 * p) = f(x_k)
        phi_p_0 = np.dot(gradient, p)  # phi'(0) = \nabla f(x_k + 0 * p_k) * p_k = \nabla f(x_k) * p_k
        alpha = 1.0

        if not phi_p_0 < 0:
            raise Exception("Expected phi'(0) < 0 to be a descent direction. but is phi'(0) =", phi_p_0)

        while alpha > 1e-16:
            _, phi_alpha = self.phi_alpha(alpha, data, lossObject, p, targets, regularization)
            if phi_alpha <= phi_0 + c_1 * alpha * phi_p_0:
                # Armijo condition satisfied
                return alpha
            alpha = alpha * theta  # theta < 1, decrease alpha
        print "backtracking line search - alpha too small"
        return -1

    def armijo_wolfe_line_search(self, c_1, c_2, data, gradient, loss, lossObject, p, targets, theta, regularization):
        """
        Performs a line search along the descent direction p, that satisfies the strong Wolfe conditions.
        :param c_1:
        :param c_2:
        :param data:
        :param gradient:
        :param loss:
        :param lossObject:
        :param p:
        :param targets:
        :param theta:
        :param regularization:
        :return: step length satisfying the strong Wolfe conditions or -1 if the maximum number of
                 iterations is reached.
        """
        # phi(alpha) = f(x_k + alpha * p_k)
        phi_0 = loss  # phi(0) = f(x_k + 0 * p) = f(x_k)
        phi_p_0 = np.dot(gradient, p)  # phi'(0) = \nabla f(x_k + 0 * p_k) * p_k = \nabla f(x_k) * p_k

        if not phi_p_0 < 0:
            raise Exception("Expected phi'(0) < 0 to be a descent direction. but is phi'(0) =", phi_p_0)

        max_iter = 200
        alpha_i = 1.0
        alpha_old = 0.0  # alpha_0 = 0

        for i in range(max_iter):
            # 1. evaluate phi(alpha_i)
            gradient_alpha_i, phi_alpha_i = self.phi_alpha(alpha_i, data, lossObject, p, targets, regularization)

            # 2. if phi(alpha_i) > phi(0) + c1 * alpha_i * phi'(0) or [phi(alpha_i) >= phi(alpha_{i-1}) and i > 0]
            if phi_alpha_i > phi_0 + c_1 * alpha_i * phi_p_0 or (i > 0 and phi_alpha_i >= phi_alpha_old):
                alpha_star = self.zoom(alpha_old, alpha_i, p, phi_0, phi_p_0, c_1, c_2, data, targets, lossObject, regularization)
                return alpha_star

            # 3. evaluate phi'(alpha_i) = \nabla f(x_k + alpha_i * p_k) * p_k
            phi_p_alpha_i = np.dot(gradient_alpha_i, p)

            # 4. if |phi'(alpha_i)| <= - c_2 * phi'(0) (strong Wolfe satisfied?)
            if abs(phi_p_alpha_i) <= - c_2 * phi_p_0:
                alpha_star = alpha_i
                return alpha_star

            # 5. if phi'(alpha_i) >= 0 (if the derivative is positive)
            if phi_p_alpha_i >= 0:
                alpha_star = self.zoom(alpha_i, alpha_old, p, phi_0, phi_p_0, c_1, c_2, data, targets, lossObject, regularization)
                return alpha_star

            # save previous results and iterate
            alpha_old = alpha_i
            phi_alpha_old = phi_alpha_i

            # 6. choose alpha_{i+1}
            alpha_i = alpha_i / theta
            #alpha_i = self.interpolate(alpha_i, 1 + alpha_i, data, lossObject, p, targets, regularization)

        print "line search/bracketing phase - max iterations"
        return -1

    def zoom(self, alpha_low, alpha_high, p, phi_0, phi_p_0, c_1, c_2, data, targets, lossObject, regularization):
        """
        Interpolation/bisection phase of A-W line search. Returns a point between alpha_low and alpha_high
        that satisfies the strong wolfe conditions. If the maximum number of iterations is reached it returns -1.

        :param alpha_low: lower bound of interval step size
        :param alpha_high: upper bound of interval step size
        :param p: search (descent) direction
        :param phi_0: phi(0)
        :param phi_p_0: phi'(0)
        :param c_1: sufficient decrease condition parameter
        :param c_2: curvature condition parameter
        :param data:
        :param targets:
        :param lossObject:
        :param regularization:
        :return:
        """
        max_iter = 200

        for i in range(max_iter):
            # 1. interpolate to find a step trial alpha_low < alpha_j < alpha_high
            #alpha_j = self.interpolate(alpha_low, alpha_high, data, lossObject, p, targets, regularization)
            #alpha_j = self.safeguarded_interpolation(alpha_high, alpha_low, 0.01, data, lossObject, p, targets, regularization)
            alpha_j = select_random_point_between(alpha_low, alpha_high)
            # 2. evaluate phi(alpha_j)
            gradient_alpha_j, phi_alpha_j = self.phi_alpha(alpha_j, data, lossObject, p, targets, regularization)

            # evaluate phi(alpha_low)
            _, phi_alpha_low = self.phi_alpha(alpha_low, data, lossObject, p, targets, regularization)

            # 3. if phi(alpha_j) > phi(0) + c_1 * alpha_j * phi'(0) or phi(alpha_j) >= phi(alpha_low)
            if phi_alpha_j > phi_0 + c_1 * alpha_j * phi_p_0 or phi_alpha_j >= phi_alpha_low:
                alpha_high = alpha_j
            else:
                # 4. evaluate phi'(alpha_j)
                phi_p_alpha_j = np.dot(gradient_alpha_j, p)
                # 5. if |phi'(alpha_j)| <= - c_2 * phi'(0) (strong Wolfe satisfied?)
                if abs(phi_p_alpha_j) <= - c_2 * phi_p_0:
                    return alpha_j
                # 6. if phi'(alpha_j)(alpha_high - alpha_low) >= 0
                if phi_p_alpha_j * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
                alpha_low = alpha_j

            if abs(alpha_high - alpha_low) < 1e-16:
                print "zoom - interval too small"
                return -1

        print "zoom - max iterations"
        return -1

    def interpolate(self, alpha_low, alpha_high, data, lossObject, p, targets, reg):
        """
        find a trial step alpha_j between alpha_low and alpha_high by quadratic interpolation.
        :param alpha_high: left edge of the interval containing step sizes satisfying the wolfe conditions
        :param alpha_low: right edge of the interval containing step sizes satisfying the wolfe conditions
        :param data: dataset
        :param lossObject: object used to compute the loss and its derivative
        :param p: descent direction
        :param targets: target variables of the patterns in the dataset
        :return:
        """
        # 1.1 evaluate phi(alpha_low), phi'(alpha_low), and phi(alpha_high)
        gradient_alpha_low, phi_alpha_low = self.phi_alpha(alpha_low, data, lossObject, p, targets, reg)
        phi_p_alpha_low = np.dot(gradient_alpha_low, p)
        _, phi_alpha_high = self.phi_alpha(alpha_high, data, lossObject, p, targets, reg)
        # 1.2 interpolate
        #alpha_j = - (phi_p_alpha_low * alpha_high ** 2) / \
        #          (2 * (phi_alpha_high - phi_alpha_low - phi_p_alpha_low * alpha_high))
        #return alpha_j
        alpha_min = alpha_low - 0.5 * (alpha_low - alpha_high) * phi_p_alpha_low / \
        (phi_p_alpha_low - (phi_alpha_low - phi_alpha_high) / (alpha_low - alpha_high))
        return alpha_min

    def safeguarded_interpolation(self, alpha_high, alpha_low, sfgrd, data, lossObject, p, targets, reg):
        """
        find a trial step size alpha_j between alpha_low and alpha_high by safeguarded quadratic interpolation
        between function values phi(alpha_low) and phi(alpha_high).
        :param alpha_high:
        :param alpha_low:
        :param sfgrd:
        :param data:
        :param lossObject:
        :param p:
        :param targets:
        :return:
        """
        gradient_alpha_low, phi_alpha_low = self.phi_alpha(alpha_low, data, lossObject, p, targets, reg)
        gradient_alpha_high, phi_alpha_high = self.phi_alpha(alpha_high, data, lossObject, p, targets, reg)
        phi_p_alpha_low = np.dot(gradient_alpha_low, p)
        phi_p_alpha_high = np.dot(gradient_alpha_high, p)

        a = (alpha_low * phi_p_alpha_high - alpha_high * phi_p_alpha_low) / (phi_p_alpha_high - phi_p_alpha_low)
        first = alpha_low * (1 + sfgrd)
        second = min(alpha_high * (1 - sfgrd), a)
        alpha_j = max(first, second)
        return alpha_j

    def phi_alpha(self, alpha_i, data, lossObject, p, targets, regularization):
        """
        Computes phi(alpha) = f(x_k + alpha_i * p_k), where
        - x_k are the current weights of the network
        - alpha_i is the trial step_size
        - p_k is the descent direction
        - f is the function to minimize (i.e the loss)
        :param alpha_i: trial step size
        :param data: dataset
        :param lossObject: object to compute the loss and its derivative
        :param p: descent direction
        :param targets: target variables of the patterns in the dataset
        :param regularization
        :return:
            - gradient_alpha = nabla f(x_k + alpha_i * p_k)
            - loss_alpha     = phi(alpha_i)
        """
        # compute x_{k+1} = x_k + alpha * p_k, and evaluates phi(alpha_i) = loss
        self.update_weights_BFGS(delta=alpha_i * p)
        gradient_alpha, loss_alpha, _ = self.calculate_gradient(data, targets, lossObject, regularization)
        # restore weights
        self.update_weights_BFGS(delta=-alpha_i*p)

        return gradient_alpha, loss_alpha

    # ------------------- end BFGS & L-BFGS ----------------------------------

    def predict(self, data):
        """
        predict target variables, returns and array, where each element is an array of scores.
        :param data: dataset which targets have to be predicted
        :return: scores
        """
        scores = []
        for pattern in data:
            self.forward(pattern)
            scores.append(self.getOutput())
        return scores

    def dump_weights(self, file_output=None):
        """
        Dump neural network weights to file, calls dump_weights() on each layer
        :param file_output: file to dump the weights to
        :return:
        """
        file_output = sys.stdout if file_output is None else file_output
        # the first line is the architecture
        if file_output == sys.stdout:
            print self.architecture
        else:
            np.save(file_output, self.architecture)
        # subsequent lines are the weights
        for layer in self.layers[1:]:  # exclude input layer
            layer.dump_weights(file_output)

    def load_weights(self, file_input):
        """
        Load weights of a neural network from a file. Raises an exception if the architecture does not match.
        :param file_input: file to read the weights from
        :return:
        """
        architecture = np.load(file_input)

        if not np.array_equal(architecture, self.architecture):
            raise Exception("The network architectures do not match: "
                            "expected " + str(self.architecture) +
                            "\ngiven " + str(architecture))

        # TODO maybe also check the type of neurons

        for layer in self.layers[1:]:  # skip input layer
            for neuron in layer.neurons[:-1]:  # skip bias neuron
                neuron.weights = np.load(file_input)

    def plot_phi_alpha_in_neighborhood(self, data, lossObject, p, targets, regularization):
        alpha_values = np.linspace(0, 5e-10, 500)
        phi_values = []
        for a in alpha_values:
            gradient_alpha_j, phi_alpha_try = self.phi_alpha(a, data, lossObject, p, targets, regularization)
            phi_values.append(phi_alpha_try)

        plt.figure()
        plt.plot(alpha_values, phi_values)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\phi(\alpha)$')
        plt.legend(loc='best')
        #plt.yticks([round(min(phi_values),2), round(max(phi_values),2)])
        plt.yscale('log')
        #plt.xscale('log')
        plt.show()

    def plot_phi_alpha_and_tangent_line(self, data, lossObject, targets, p, gradient, regularization):
        alpha_values = np.linspace(0, 1, 500)
        phi_values = []

        # phi(alpha)
        for a in alpha_values:
            gradient_alpha_j, phi_alpha_try = self.phi_alpha(a, data, lossObject, p, targets, regularization)
            phi_values.append(phi_alpha_try)

        # tangent line at alpha=0
        x = np.linspace(0, 1, 10)
        phi_0 = phi_values[0]
        phi_p_0 = np.dot(gradient, p)
        y = phi_p_0 * x + phi_0 # y = mx + q

        plt.figure()
        plt.plot(alpha_values, phi_values, label=r'$\phi(\alpha)$')
        plt.plot(x, y, label='retta tangente a ' + r'$\phi(0)$', color='red')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\phi(\alpha)$')
        plt.legend(loc='best')
        plt.yscale('log')
        #plt.xscale('log')
        plt.show()


def check_topology(architecture, neurons):
    """
    Checks that the newly created neural network has a proper architecture
    (i.e. the neurons of the first layer are InputNeurons ecc..)
    :param architecture: network architecture
    :param neurons: list of layers' neurons' type
    :return:
    """
    if len(architecture) != len(neurons):
        raise Exception("Architecture miss match")
    if not neurons[0].__name__ is InputNeuron.__name__:
        raise Exception("Input neurons have incorrect type")
    for i in range(1, len(neurons) - 1):
        if neurons[i].__name__ is InputNeuron.__name__ or neurons[i].__name__ is OutputNeuron.__name__:
            raise Exception("Hidden neurons have incorrect type")


def select_random_point_between(alpha_low, alpha_high):
    """
    select a trial step size alpha_j between alpha_low and alpha_high randomly.
    :param alpha_low:
    :param alpha_high:
    :return:
    """
    convex = 0.5  # bisection
    alpha_j = convex * alpha_low + (1 - convex) * alpha_high
    return alpha_j
