from Layer import *
from loss_functions import *
import sys
from collections import deque
from scipy.linalg import norm
import random


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

    def shuffle_dataset(self,data,targets):
        permumation = np.random.permutation(len(data))
        data_shuffled = [data[i] for i in permumation]
        targets_shuffled = [targets[i] for i in permumation]
        return data_shuffled,targets_shuffled

    def intialize_weight(self):
        """
        reinitialize network's wights (usefule in grid search?)
        :return:
        """
        for l in range(1,len(self.layers)):
            for n in range(0,len(self.layers[l].neurons)-1):
                len_weights = len(self.layers[l].neurons[n].weights)
                self.layers[l].neurons[n].weights = np.random.uniform(low=-0.7, high=0.7, size=len_weights)

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

    def back_propagation(self, target, lossObject):
        """
        Performs backpropagation.

        :param target: target vector for a single training example
        :param lossObject: function to optimize
        :return: gradient_weights, gradient w.r.t network weights
                 loss_value, loss value computed by lossObject
                 misClassification, misclassification error
        """
        # vector containing the deltas of each layer in reverse order
        # (i.e from output layer to first hidden layer)
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
        gradient_weights = self.compute_gradient(np.asarray(delta_vectors))

        # 5 report loss and misclassification count
        weights = self.get_weights_as_vector()

        loss_value = lossObject.value(target, output_net, weights)
        misClassification = lossObject.misClassification(target, output_net)

        return gradient_weights, loss_value, misClassification


    def get_weights_as_vector(self):
        """
        gets neural network weights as a single array.
        :return: array of weights
        """
        weights = np.array([neuron.weights for layer in self.layers[1:] for neuron in layer.neurons[:-1]])
        weights = np.concatenate(weights).ravel()
        return weights


    def compute_gradient(self, delta_vectors):
        """
        Computes the gradient from the delta of each neuron.

        :param delta_vectors: vector where each entry delta_vectors[l][n]
                contains the delta of each neuron 'n' in layer 'l'.
        :return: a matrix gradient_w where each entry gradient[l][n][w] is the gradient w.r.t
                the weight 'w' of the neuron 'n' in the layer 'l'.
        """
        gradient_w = []
        for l in range(1, len(self.layers)):
            tmpL = []
            for n in range(len(self.layers[l].neurons[:-1])):  # exclude bias neuron
                neuron_input = np.asarray([neuron.getOutput() for neuron in self.layers[l-1].neurons])
                tmpL.append(neuron_input * delta_vectors[l-1][n])
            gradient_w.append(tmpL)

        return np.asarray(gradient_w)

    def update_weights(self, gradient_w, learning_rate, prev_delta, momentum, regularization=0):
        """
        update weights as
            DeltaW_{ji} = gradient_w * learning_rate - regularization*w_{ji} + momentum*prevg (old gradient)
            w_{ji}+= DeltaW_{ji}

        :param gradient_w: gradient error on data wrt neural network`s weights
        :param learning_rate: learning rate
        :param prev_delta: previous delta for momentum
        :param momentum: percentage of prev_delta
        :param regularization: regularization coefficient
        :return: current current weight update (i.e prev_delta for next iteration)
        """
        deltaW = - learning_rate * gradient_w + momentum * prev_delta
        # initialize a vector of deltaw`s shape
        lambda_vectors = copy.deepcopy(deltaW)
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons) - 1):
                # fill vectors with regularization coefficient, 0 in bias`s entry
                lambda_vectors[i-1][j] = regularization
                lambda_vectors[i-1][j][-1] = 0
                # compute regularization gradient (wrt current weights) = w*regularization_coefficient
                regularization_term = np.multiply(self.layers[i].neurons[j].weights,lambda_vectors[i-1][j])
                # add regularization gradient to total weight`s update
                deltaW[i-1][j] -= regularization_term
                # update weights
                self.layers[i].neurons[j].weights += deltaW[i-1][j]
        return deltaW

    def compute_delta_hidden_layer(self, delta_next_layer, currentLayerIndex):
        # delta_layer vector
        delta_layer = np.array([])
        for h in range(len(self.layers[currentLayerIndex].neurons)-1):
            downstream = self.layers[currentLayerIndex + 1].neurons[:-1]
            weights = [neuron.weights[h] for neuron in downstream]
            gradient_flow = np.dot(weights, delta_next_layer)
            derivative_net = self.layers[currentLayerIndex].neurons[h].activation_function_derivative()
            delta_h = gradient_flow * derivative_net
            delta_layer = np.append(delta_layer, delta_h)
        return delta_layer

    def compute_delta_output_layer(self, output_net, target, loss):
        output_layer = self.layers[-1]
        af_derivatives = np.array([neuron.activation_function_derivative() for neuron in output_layer.neurons[:-1]])
        error_derivatives = loss.derivative(np.array(target), output_net)
        delta_outputLayer = np.multiply(af_derivatives, error_derivatives)
        return delta_outputLayer

    def validation_error(self,patterns,targets,loss_obj):
        """
        compute squared and misclassification error on validation set
        :param patterns: validation set patterns
        :param targets: validation set targets
        :param loss_obj: loss for computing error (same of traning set)
        :return:
        """
        # predict each validation set pattern
        scores = self.predict(patterns)

        squared_error_epoch = 0
        misClass_error_epoch = 0
        # add errors
        for i in range(len(scores)):
            squared_error_epoch += loss_obj.value(targets[i],scores[i],[ [], [] ])
            misClass_error_epoch += loss_obj.misClassification (targets[i],scores[i])
        # return sum of a single validation epoch
        return squared_error_epoch, misClass_error_epoch

    def train(self, data, targets, eval_data, eval_targets, lossObject, epochs, learning_rate, batch_size, momentum,
              regularization=0):
        """
        Performs the training of the neural network.

        :param data: traning set patterns
        :param targets: traning set target for each pattern in 'data'
        :param vl_data : validation set patterns
        :param vl_targets: validation set targets
        :param lossObject: loss
        :param epochs:
        :param learning_rate:
        :param batch_size:
        :param momentum:
        :param regularization: regularization strength

        :return: losses, vector of the loss computed at each epoch
                 misClassification, vector of misclassification loss for each epoch
        """

        # lists for specify missclassification and Squared error (for traning and validation)
        losses = np.array([])
        misClassification = np.array([])
        losses_valdation = np.array([])
        misClassification_validation = np.array([])
        # prev_delta is previous weights update (for momentum)
        prev_delta = np.array([np.zeros((self.architecture[i], self.architecture[i - 1] + 1))
                         for i in range(1, len(self.architecture))])
        for epoch in range(epochs):
            # current epoch value of misclassification and Squared error
            loss_epoch = 0
            misC_epoch = 0
            # shuffle data set
            data_shuffled, targets_shuffled = self.shuffle_dataset(data, targets)
            
            for i in range(0, len(data_shuffled), batch_size):
                # take only batch_size examples
                batch_pattern = data_shuffled[i:i + batch_size]
                batch_target = targets_shuffled[i:i + batch_size]
                # gradient_w_batch = sum of gradient_w for the epoch
                gradient_w_batch = np.array([np.zeros((self.architecture[i], self.architecture[i - 1] + 1))
                                            for i in range(1, len(self.architecture))])
                # train, compute gradient for a batch
                for pattern, t in zip(batch_pattern, batch_target):
                    self.forward(pattern)
                    gradient_w, loss_p, miss_p = self.back_propagation(t, lossObject)
                    loss_epoch += loss_p
                    misC_epoch += miss_p
                    gradient_w_batch += gradient_w

                gradient_w_batch /= len(batch_pattern)  # take mean gradient across batch
                # update neural network weights after a batch of training example
                # save previous weight update
                prev_delta = self.update_weights(gradient_w=gradient_w_batch,
                                                 learning_rate=learning_rate,
                                                 prev_delta=prev_delta,
                                                 momentum=momentum,
                                                 regularization=regularization * len(batch_pattern) / len(data))

            # append the total loss and misClassification of single epoch
            losses = np.append(losses, loss_epoch)
            misClassification = np.append(misClassification, misC_epoch)

            # computing loss and misClassification on validation set then append to list
            squared_error_validation_epoch, misClass_error_validation_epoch = \
                self.validation_error(eval_data, eval_targets, lossObject)

            losses_valdation = np.append(losses_valdation, squared_error_validation_epoch)
            misClassification_validation = np.append(misClassification_validation, misClass_error_validation_epoch)

        return losses, misClassification, losses_valdation,misClassification_validation

#begginning CM part -------------------------------------------------------


    def get_gradient_as_vector(self,list):
        gradient = np.array([])
        for l in list:
            tmp = np.concatenate(l)
            gradient = np.append(gradient,tmp)
        return gradient

    def update_weights_CM(self, delta):
        """
        update network weights
        :param delta: weight update p_k = - H * nabla f
        :return: x_k+1
        """
        start = 0
        # initializing x_old = x_k and x_new = x_{k+1}
        x_new = np.array([])

        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons) - 1):
                current_neuron_weights = self.layers[i].neurons[j].weights
                # append to x_k before updating

                # taking only gradient's entry w.r.t. current gradient
                weights_len = len(current_neuron_weights)
                tmp = delta[start:start + weights_len]
                start += weights_len
                # update weigths
                current_neuron_weights += tmp

                # append to x_{k+1} after updating
                x_new = np.append(x_new, current_neuron_weights)
        return x_new

    def calculate_gradient(self, data, targets, lossObject):
        # create empty vector, gradient_w_old = sum of gradient_w for the epoch
        gradient_w_batch = np.array([np.zeros((self.architecture[i], self.architecture[i - 1] + 1))
                                     for i in range(1, len(self.architecture))])
        loss_batch = 0
        miss_batch = 0
        for pattern, t in zip(data, targets):
            # calculate derivative for every patten, then append to gradient_w_batch
            self.forward(pattern)
            gradient_w, loss_p, miss_p = self.back_propagation(t, lossObject)

            gradient_w_batch += gradient_w
            loss_batch += loss_p
            miss_batch += miss_p

        # getting the gradient as vector
        gradient = self.get_gradient_as_vector(gradient_w_batch)
        gradient = - gradient  # invert sign because of implementation

        # compute mean values
        gradient /= len(data)
        loss_batch /= len(data)
        miss_batch /= float(len(data))
        return gradient, loss_batch, miss_batch

    def update_matrix(self, H_k, s_k, y_k):
        # get dimension
        shape = H_k.shape[0]

        # rho_k = 1/(y_k^t*s_k)
        rho_k = float(1) / np.dot(s_k, y_k)

        # V_k = I - rho_k * s_k * y_k^t
        tmp = rho_k * np.outer(s_k, y_k)
        V_k = np.identity(shape) - tmp

        # H_{k+1} = V_k^t * H_k * V_k - rho_k * s_k * s_k^t
        tmp = np.dot(V_k.T, H_k)
        H_new = np.dot(tmp, V_k)
        # adding rho_k*s_k*s_k^t
        H_new += np.outer(s_k, s_k) * rho_k
        return H_new


    def trainBFGS(self, data, targets, eval_data, eval_targets, lossObject,epochs):
        losses = []  # vector containing the loss of each epoch
        misses = []  # vector containing the misclassification for each epoch
        # 1. compute initial gradient and initial Hessian approximation H_0
        gradient_old, loss, miss = self.calculate_gradient(data, targets, lossObject)
        H = np.identity(gradient_old.shape[0])
        x_old = self.get_weights_as_vector()
        # append losses
        losses.append(loss)
        misses.append(miss)
        print "epoch\tMSE\t\t\tmisclass\t\tnorm(g)\t\tnorm(h)\t\trho\t\t\talpha"
        print "---------------------------------------------------------------------------"
        for epoch in range(epochs):
            # stop criterion
            if epoch > 0 and (norm(gradient_old)) < 1e-6:
                print "break at", epoch
                break

            # 1. compute search direction p = -H * gradient
            p = - H.dot(gradient_old)

            # 2. line search
            theta = 0.9   # contraction factor of alpha
            alpha_0 = 1   # initial step size trial is always 1 for quasi-Newton TODO: try initial step less than 1
            c_1 = 0.0001  # scaling factor for Armijo condition TODO try 1e-4
            c_2 = 0.9     # scaling factor for Wolfe condition

            #alpha = self.backtracking_line_search(alpha_0, c_1, data, epoch, gradient_old, loss, lossObject, p, targets, theta)
            alpha = self.armijo_wolfe_line_search(alpha_0, c_1, c_2, data, gradient_old, loss, lossObject, p, targets, theta)

            # 3. compute weight update
            delta = p * alpha

            # 4. update weights using x_{k+1} = x_{k} + alpha_{k} * p_k
            x_new = self.update_weights_CM(delta)

            # 5. compute new gradient
            gradient_new, loss, miss = self.calculate_gradient(data, targets, lossObject)

            # append losses
            losses.append(loss)
            misses.append(miss)

            # 6. compute s_k = x_{k+1} - x_k = x_new - x_old
            # compute y_k = nabla f_{k+1} - nabla f_k = gradient new - gradient old
            s_k = x_new - x_old
            y_k = gradient_new - gradient_old

            # 7. update matrix H
            H = self.update_matrix(H, s_k, y_k)

            # print statistics
            print "%d\t\t%f\t%f\t\t%f\t%f\t%f\t%f" % \
                  (epoch+1, loss, miss, norm(gradient_new), norm(H), float(1)/np.dot(s_k, y_k), alpha)

            # update x_old and gradient_old
            x_old = x_new
            gradient_old = gradient_new

        return losses, misses

    def compute_direction(self,H,gradient,s,y,rho):

        a_list = []
        q = gradient
        # first loop
        for i in range(len(s)- 1, -1 , -1):
            a = rho [i]*np.dot(s[i],q)
            a_list.insert(0,a)
            q -= -a*y[i]

        # da wikipedia prossime due righe
        #if not len(s)==0:
         #   H = np.outer(y[0], s[0]) / np.dot(y[0],y[0])
        r = H.dot(q)

        #second loop
        for i in range(len(s)):
            beta = rho[i]*np.dot(y[i],r)
            r +=  s[i]* (a_list[i]-beta)

        return r


    def trainLBFGS (self, data, targets, eval_data, eval_targets, lossObject,m,epochs):

        losses = []  # vector containing the loss of each epoch
        misses = []  # vector containing the misclassification for each epoch
        # 1. compute initial gradient and initial Hessian approximation H_0
        gradient_old, loss, miss = self.calculate_gradient(data, targets, lossObject)
        H = np.identity(gradient_old.shape[0])
        x_old = self.get_weights_as_vector()

        # set of current s,y,p
        s_list = []
        y_list = []
        rho_list = []

        # append losses
        losses.append(loss)
        misses.append(miss)
        print "epoch\tMSE\t\t\tmisclass\t\tnorm(g)\t\tnorm(h)\t\trho\t\t\talpha"
        print "---------------------------------------------------------------------------"
        for epoch in range(epochs):
            # stop criterion
            if epoch > 0 and (norm(gradient_old)) < 1e-6:
                print "break at", epoch
                break



            # compute p using two loop recursion
            r= self.compute_direction(H,gradient_old,s_list,y_list,rho_list)
            p = -r

            # 2. line search
            theta = 0.9   # contraction factor of alpha
            alpha_0 = 1   # initial step size trial is always 1 for quasi-Newton TODO: try initial step less than 1
            c_1 = 0.0001  # scaling factor for Armijo condition TODO try 1e-4
            c_2 = 0.9     # scaling factor for Wolfe condition
            alpha = self.armijo_wolfe_line_search(alpha_0, c_1, c_2, data, gradient_old, loss, lossObject, p, targets, theta)
            #print "alpha is", alpha

            # updating weights and compute x_k+1 = x_k + a_k*p_k
            delta = p*alpha
            x_new = self.update_weights_CM(delta)
            gradient_new, loss,miss = self.calculate_gradient(data,targets,lossObject)


            #print "loss is", loss

            if epoch>(m-1):
                #discard first element
                del s_list[0]
                del y_list[0]
                del rho_list[0]

            # compute s_k , y_k, p_k
            s_k = x_new - x_old
            y_k = gradient_new - gradient_old
            rho_k = 1/ np.dot(s_k,y_k)

#            H = self.update_matrix()

            #append to vector
            s_list.append(s_k)
            y_list.append(y_k)
            rho_list.append(rho_k)

            # print statistics
            print "%d\t\t%f\t%f\t\t%f\t%f\t%f\t%f" % \
                  (epoch+1, loss, miss, norm(gradient_new), norm(H), float(1)/np.dot(s_k, y_k), alpha)

            #udate x_old and gragient_olf
            x_old = x_new
            gradient_old = gradient_new

        return losses, misses


    def backtracking_line_search(self, alpha, c_1, data, epoch, gradient_old, loss, lossObject, p, targets, theta):
        while True:
            _, phi_alpha = self.evaluate_phi_alpha(alpha, data, lossObject, p, targets)
            phi_0 = loss                       # phi(0) = f(x_k + 0 * p) = f(x_k)
            phi_p_0 = np.dot(gradient_old, p)  # phi'(0) = \nabla f(x_k + 0 * p_k) * p_k = \nabla f(x_k) * p_k

            if phi_alpha <= phi_0 + c_1 * alpha * phi_p_0:
                # Armijo condition satisfied
                break
            if alpha < 1e-16:
                print "some error in the algorithm. alpha:", alpha
                break

            alpha *= theta  # theta < 1, decrease alpha

        return alpha

    def armijo_wolfe_line_search(self, alpha, c_1, c_2, data, gradient_old, loss, lossObject, p, targets, theta):
        # phi(alpha) = f(x_k + alpha * p_k)
        # phi'(alpha) = \nabla f(x_k + alpha * p_k) * p_k
        alpha_max = 10
        alpha_i = alpha  # alpha_1 > 0
        alpha_old = 0    # alpha_0
        default_alpha = 0.01  # step to take if there was an error in the line search (returned alpha less than 1e-16)
        i = 1
        while True:
            # 1. evaluate phi(alpha_i)
            gradient_alpha_i, loss_alpha_i = self.evaluate_phi_alpha(alpha_i, data, lossObject, p, targets)
            phi_alpha_i = loss_alpha_i

            # 2. if phi(alpha_i) > phi(0) + c1 * alpha_i * phi_p(0) or [phi(alpha_i) >= phi(alpha_{i-1}) and i > 1]
            phi_0 = loss                       # phi(0) = f(x_k + 0 * p) = f(x_k)
            phi_p_0 = np.dot(gradient_old, p)  # phi'(0) = \nabla f(x_k + 0 * p_k) * p_k = \nabla f(x_k) * p_k

            if not phi_p_0 < 0:
                raise Exception("Expected phi'(0) < 0 to be a descent direction. but is phi'(0) =", phi_p_0)

            if phi_alpha_i > phi_0 + c_1 * alpha_i * phi_p_0 or (i > 1 and phi_alpha_i >= phi_alpha_old):
                alpha_star = self.zoom(alpha_old, alpha_i, p, phi_0, phi_p_0, c_1, c_2, data, targets, lossObject)
                break

            # 3. evaluate phi'(alpha_i)
            phi_p_alpha_i = np.dot(gradient_alpha_i, p)

            # 4. if |phi'(alpha_i)| <= - c_2 * phi'(0) (strong Wolfe satisfied?)
            if abs(phi_p_alpha_i) <= - c_2 * phi_p_0:  # TODO try with c_2 * |phi'(0)| or with frangioni formulae
                alpha_star = alpha_i
                break

            # 5. if phi'(alpha_i) >= 0 (if the derivative is positive)
            if phi_p_alpha_i >= 0:
                alpha_star = self.zoom(alpha_i, alpha_old, p, phi_0, phi_p_0, c_1, c_2, data, targets, lossObject)
                break

            # 6. choose alpha_{i+1} in (alpha_i, alpha_max)
            tmp_alpha = alpha_i / theta
            alpha_i = tmp_alpha if tmp_alpha < alpha_max else alpha_max

            # save previous results and iterate
            alpha_old = alpha_i
            phi_alpha_old = phi_alpha_i
            i += 1

        if alpha_star <= 1e-16:
            print "error, alpha =", alpha_star, "set alpha =", default_alpha
            alpha_star = default_alpha

        return alpha_star

    def zoom(self, alpha_low, alpha_high, p, phi_0, phi_p_0, c_1, c_2, data, targets, lossObject):
        max_feval = 100
        sfgrd = 0.1

        for i in range(max_feval):
            # 1. interpolate to find a step trial alpha_low < alpha_j < alpha_high
            #alpha_j = self.interpolate(alpha_high, alpha_low, data, lossObject, p, targets)
            #alpha_j = self.safeguarded_interpolation(alpha_high, alpha_low, sfgrd, data, lossObject, p, targets)
            alpha_j = select_random_point_between(alpha_low, alpha_high)

            # 2. evaluate phi(alpha_j)
            gradient_alpha_j, loss_alpha_j = self.evaluate_phi_alpha(alpha_j, data, lossObject, p, targets)
            phi_alpha_j = loss_alpha_j

            # 3. if phi(alpha_j) > phi(0) + c_1 * alpha_j * phi'(0) or phi(alpha_j) >= phi(alpha_low)
            # evaluate phi(alpha_low)
            _, loss_alpha_low = self.evaluate_phi_alpha(alpha_low, data, lossObject, p, targets)
            phi_alpha_low = loss_alpha_low
            if phi_alpha_j >= phi_0 + c_1 * alpha_j * phi_p_0 or phi_alpha_j >= phi_alpha_low:
                alpha_high = alpha_j
            else:
                # 4. evaluate phi'(alpha_j)
                phi_p_alpha_j = np.dot(gradient_alpha_j, p)
                # 5. if |phi'(alpha_j)| <= - c_2 * phi'(0) (Wolfe satisfied?)
                # if abs(phi_p_alpha_j) <= c_2 * abs(phi_p_0):  # strong wolfe
                # if phi_p_alpha_j >= c_2 * phi_p_alpha_j:  # wolfe frangio
                if abs(phi_p_alpha_j) <= - c_2 * phi_p_0:  # book: strong wolfe
                    return alpha_j
                # 6. if phi'(alpha_j)(alpha_high - alpha_low) >= 0
                if phi_p_alpha_j * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
                alpha_low = alpha_j

        print "max zoom iterarions"
        return alpha_j

    def interpolate(self, alpha_high, alpha_low, data, lossObject, p, targets):
        """
        find a trial step alpha_j between alpha_low and alpha_high by quadratic interpolation.

        :param alpha_high: left edge of the interval containing step sizes satisfying the wolfe condition
        :param alpha_low: right edge of the interval containing step sizes satisfying the wolfe condition
        :param data: dataset
        :param lossObject: object used to compute the loss and its derivative
        :param p: direction of descent
        :param targets: target variables of the patterns in the dataset
        :return:
        """
        # 1.1 evaluate phi(alpha_low), phi'(alpha_low), and phi(alpha_high)
        gradient_alpha_low, phi_alpha_low = self.evaluate_phi_alpha(alpha_low, data, lossObject, p, targets)
        phi_p_alpha_low = np.dot(gradient_alpha_low, p)
        _, phi_alpha_high = self.evaluate_phi_alpha(alpha_high, data, lossObject, p, targets)
        # 1.2 interpolate
        alpha_j = - (phi_p_alpha_low * alpha_high ** 2) / \
                  (2 * (phi_alpha_high - phi_alpha_low - phi_p_alpha_low * alpha_high))
        return alpha_j

    def safeguarded_interpolation(self, alpha_high, alpha_low, sfgrd, data, lossObject, p, targets):
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
        gradient_alpha_low, phi_alpha_low = self.evaluate_phi_alpha(alpha_low, data, lossObject, p, targets)
        gradient_alpha_high, phi_alpha_high = self.evaluate_phi_alpha(alpha_high, data, lossObject, p, targets)
        phi_p_alpha_low = np.dot(gradient_alpha_low, p)
        phi_p_alpha_high = np.dot(gradient_alpha_high, p)

        a = (alpha_low * phi_p_alpha_high - alpha_high * phi_p_alpha_low) / (phi_p_alpha_high - phi_p_alpha_low)
        first = alpha_low * (1 + sfgrd)
        second = min(alpha_high * (1 - sfgrd), a)
        alpha_j = max(first, second)
        return alpha_j


    def evaluate_phi_alpha(self, alpha_i, data, lossObject, p, targets):
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
        :return:
            - gradient_alpha = nabla f(x_k + alpha_i * p_k)
            - loss_alpha     = phi(alpha_i)
        """
        # creates a copy of the network, update its weights to get the
        # hypothetical x_{k+1} = x_k + alpha * p_k, and evaluates phi(alpha_i) = loss
        temp_network = copy.deepcopy(self)
        temp_network.update_weights_CM(alpha_i * p)
        gradient_alpha, loss_alpha, _ = temp_network.calculate_gradient(data, targets, lossObject)
        return gradient_alpha, loss_alpha

    # end CM-----------------------------------------------------------

    def predict(self, data):
        # predict target variables
        # returns and array, where each element is an array of #output scores.
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

        """ TODO maybe also check the type of neurons? (e.g. if the network that was trained
            had Sigmoids, should we raise an error if the network we want to load has TanH. I think so.
        """
        for layer in self.layers[1:]:  # skip input layer
            for neuron in layer.neurons[:-1]:  # skip bias neuron
                neuron.weights = np.load(file_input)


def check_topology(architecture, neurons):
    """
    Checks that the newly created neural network has a proper architectures
    (i.e. the neurons of the first layer are InputNeurons)

    :param architecture: network architecture
    :param neurons: list of layers' neurons' type
    :return:
    """
    if len(architecture) != len(neurons):
        raise Exception("Architecture miss match")
    if not neurons[0].__name__ is InputNeuron.__name__:
        raise Exception("Input neurons have incorrect type")
    #if not neurons[-1].__name__ is OutputNeuron.__name__:
    #    raise Exception("Output neurons have incorrect type")
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
    convex = random.uniform(0.1, 0.9)
    alpha_j = convex * alpha_low + (1 - convex) * alpha_high
    return alpha_j
