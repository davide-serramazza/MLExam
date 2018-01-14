from Layer import *
from loss_functions import *
import sys
from collections import deque


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
        weights = np.asarray([neuron.weights
                              for layer in self.layers[1:]
                              for neuron in layer.neurons[:-1]])

        loss_value = lossObject.value(target, output_net, weights)
        misClassification = lossObject.misClassification(target, output_net)

        return gradient_weights, loss_value, misClassification

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

    def update_weights(self, delta_w, learning_rate, regularization=0):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons) - 1):
                # add regularization (do not regularize bias weights)
                temp = self.layers[i].neurons[j].weights[:-1]  # weights to regularize, exclude weight of bias neuron
                lambda_vector = np.empty(temp.shape)
                lambda_vector.fill(regularization)
                self.layers[i].neurons[j].weights[:-1] -= np.multiply(lambda_vector, temp)
                # add gradient
                self.layers[i].neurons[j].weights += learning_rate * delta_w[i - 1][j]

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
        scores = self.predict(patterns)
        squared_error = []
        misClass_error = []
        for i in range(len(scores)):
            squared_error.append( loss_obj.value(targets[i],scores[i][0],[ [], [] ]) )
            misClass_error.append( loss_obj.misClassification (targets[i],scores[i]) )
        return np.sum(squared_error), np.sum(misClass_error)

    def train(self, data, targets, lossObject, epochs, learning_rate, batch_size, momentum, regularization=0):
        """
        Performs the training of the neural network.

        :param data: patterns
        :param targets: target for each pattern in 'data'
        :param lossObject: loss
        :param epochs:
        :param learning_rate:
        :param batch_size:
        :param momentum:
        :param regularization: regularization strength

        :return: losses, vector of the loss computed at each epoch
                 misClassification, vector of misclassification loss for each epoch
        """
        # lists for specify missclassification and Squared error
        losses = np.array([])
        misClassification = np.array([])
        # prevg is previous gradient  (for momentum)
        prevg = []
        for epoch in range(epochs):
            # current epoch value of misclassification and Squared error
            loss_epoch = 0
            misC_epoch = 0
            for i in range(0, len(data), batch_size):
                # take only batch_size examples
                batch_pattern = data[i:i + batch_size]
                batch_target = targets[i:i + batch_size]

                # gradient_w_epoch = sum of gradient_w for the epoch
                gradient_w_epoch = np.array([np.zeros((self.architecture[i], self.architecture[i - 1] + 1))
                                            for i in range(1, len(self.architecture))])
                # now really train
                for pattern, t in zip(batch_pattern, batch_target):
                    self.forward(pattern)
                    gradient_w, loss_p, miss_p = self.back_propagation(t, lossObject)

                    loss_epoch += loss_p
                    misC_epoch += miss_p
                    gradient_w_epoch += gradient_w
                # add momentum from the second epoch onwards
                if epoch == 0:
                    prevg = copy.deepcopy(gradient_w_epoch)
                else:
                    tmp = copy.deepcopy(gradient_w_epoch)   # save gradient at current epoch
                    gradient_w_epoch += prevg * momentum    # add momentum
                    prevg = tmp                          # store previous gradient

                # update neural network weights
                self.update_weights(gradient_w_epoch, learning_rate/batch_size, regularization * batch_size / len(data))

            # append the total loss and misClassification of single epoch
            losses = np.append(losses, loss_epoch)
            misClassification = np.append(misClassification, misC_epoch)

        return losses, misClassification

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