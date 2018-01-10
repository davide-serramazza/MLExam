from Layer import *
from loss_functions import *
import sys


class Network:
    def __init__(self, architecture, neurons):
        check_topology(architecture, neurons)
        self.layers = []
        self.architecture = architecture  # for quick access when writing weights to file

        # input layer
        inputNeuron = neurons[0](0)     #0 because i have to specify weight's vector lenght
        layer = Layer(architecture[0], 0, inputNeuron)
        self.layers.append(layer)
        # hidden and output layers
        for i in range(1, len(architecture)):
            len_weights = len(self.layers[i-1].neurons)
            neuron = neurons[i](len_weights=len_weights)
            layer = Layer(architecture[i], architecture[i - 1], neuron)
            self.layers.append(layer)

    def getOutput(self):
        last_layer = self.layers[-1]
        return last_layer.getOutput()[:-1]

    def forward(self, pattern):
        self.feed_input_neurons(pattern)
        self.propagate_input()
        return self.getOutput()

    def propagate_input(self):
        # propagate result
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for neuron in layer.neurons[:-1]:  # exclude bias
                weights = neuron.weights
                input_x = self.layers[i - 1].getOutput()
                scores = np.dot(weights, input_x)
                neuron.activation_function(scores)

    def feed_input_neurons(self, data):
        input_layer = self.layers[0]
        for input_neuron, x in zip(input_layer.neurons[:-1], data):  # exclude bias
            input_neuron.activation_function(x)

    def back_propagation(self, target, eta=0.1):
        # TODO: add parameter loss
        loss = SquaredError("s") #todo init to a generic object
        # understand if is needed only squared error or misclassification too i.e task = Classification or regression?
        if isinstance(self.layers[-1].neurons[0],SigmoidNeuron):
            loss = SquaredError("s")
        if isinstance(self.layers[-1].neurons[0],TanHNeuron):
            loss = SquaredError("t")
        # 1. get the output vector from the forward step
        output_net = np.array(self.getOutput())
        # propagate the errors backward through the network:
        # 2. for each network output unit compute its error term delta
        delta_output = self.compute_delta_output_units(output_net, target, loss)
        # 3. for each hidden unit compute its error term delta
        #intialize a vector to contain delta found for each layer(in reverse order)
        delta_vectors = []
        delta_vectors.append(delta_output)
        #delta next_layer = temp variable containing next layer's delta
        delta_next_layer = delta_output
        # compute delta vector for each layer and append it to result
        for hidden_layer_index in range(len(self.layers) - 2, 0, -1):
            delta_next_layer = self.compute_delta_hidden_units(delta_next_layer,hidden_layer_index)
            delta_vectors.append(delta_next_layer)
        # 4. update network weights
        # array 3d che contiene i cambiamenti da apportare ai pesi, in particolare delta_w[i][j][k] contiene
        # i cambiamenti da apportare nel layer i+1 (no modifiche ad input layer), neurone j, peso k
        delta_w = self.compute_weight_update(delta_vectors, eta)
        # 5 report loss and missclassification count
        #weights =
        loss_value = loss.value(target, output_net)
        misClassification = loss.misClassification(target,output_net)
        return delta_w, loss_value, misClassification

    def compute_weight_update(self, delta_vectors, eta):
        delta_w = []
        for i in range(1, len(self.layers)):
            tmpL = []
            for j in range(len(self.layers[i].neurons) - 1):
                tmpN = np.array([])
                for w in range(len(self.layers[i].neurons[j].weights)):
                    # qui errore precedente, ad ogni passo il neuronre di cui si prendere l'output
                    # e diverso, tuo codice aveta ...neurons[j] , adesso ...neuron[w].
                    tmp = np.array(eta * self.layers[i - 1].neurons[w].output * delta_vectors[-i][j])
                    tmpN = np.append(tmpN,tmp)
                tmpL.append(tmpN)
            tmpL = np.asarray(tmpL)
            delta_w.append(tmpL)
        delta_w = np.asarray(delta_w)
        return delta_w

    def compute_delta_hidden_units(self, delta_next_layer,i):
        #delta_layer vector
        delta_layer = []
        for h in range(len(self.layers[i].neurons)-1):
            downstream = self.layers[i+ 1].neurons[:-1]
            weights = [neuron.weights[h] for neuron in downstream]
            gradient_flow = np.dot(weights, delta_next_layer)
            d_net = self.layers[i].neurons[h].activation_function_derivative()
            delta_h = gradient_flow * d_net
            delta_layer.append(delta_h)
        return delta_layer

    def compute_delta_output_units(self, output_net, target, loss):
        output_layer = self.layers[-1]
        af_derivatives = np.array([neuron.activation_function_derivative() for neuron in output_layer.neurons[:-1]])
        diff = loss.derivative(np.array(target), output_net)
        delta_output = np.multiply(af_derivatives, diff)
        return delta_output

    def update_weights(self, delta_w, regularization=0):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons) - 1):
                self.layers[i].neurons[j].weights += delta_w[i - 1][j]
                # add regularization (do not regularize bias weights)
                temp = self.layers[i].neurons[j].weights[:-1]  # exclude weight of bias neuron
                lambda_vector = np.empty(temp.shape)
                lambda_vector.fill(regularization)
                self.layers[i].neurons[j].weights[:-1] += np.array(lambda_vector)


    def oldtrain(self, data, targets, epochs, learning_rate,l):  #, batch_size): TODO add batch size
        # fit the data
        losses = []
        for epoch in range(epochs):
            loss_batch = 0
            for pattern, target in zip(data, targets):
                self.forward(pattern)
                delta_w, loss_p = self.back_propagation(target, learning_rate,loss=l)
                loss_batch += loss_p
                self.update_weights(delta_w)
            losses.append(loss_batch)
        return losses

    def train(self, data, targets, epochs, learning_rate, batch_size, momentum, regularization=0):
        # fit the data
        # lists for specify missclassification and Squared erro
        losses = []
        misClassification = []
        # prev g is previous gradien  (for momentum)
        prevg = []
        for epoch in range(epochs):
            # current epoch vale of missclassification and Squared error
            loss_batch = 0 # TODO: loss_epoch, misC_epoch
            misC_batch = 0
            for i in range(0,len(data),batch_size):
                #take only batch_size examples
                pattern = data[i:i+batch_size]
                target = targets[i:i+batch_size]
                #deltaw_tot = sum of delata_W (delta of a single iteration)
                deltaw_Tot = []
                #now really train
                for p,t in zip (pattern,target):
                    self.forward(p)
                    delta_w, loss_p, miss_p = self.back_propagation(t, learning_rate/batch_size)
                    loss_batch += loss_p
                    misC_batch +=miss_p
                    # momenutm stuff
                    if deltaw_Tot == []:
                        deltaw_Tot=delta_w
                    else:
                        deltaw_Tot += delta_w      #non riesco ad usare np.sum...sorry
            #update weights
                if (prevg == []):
                    prevg = copy.deepcopy(deltaw_Tot)
                else:
                    tmp = copy.deepcopy(deltaw_Tot)
                    deltaw_Tot += (prevg*momentum)
                    prevg = tmp
                self.update_weights(deltaw_Tot, regularization)
            #append the total loss and missClassification in single epoch
            losses.append(loss_batch)
            misClassification.append(misC_batch)
        return losses,misClassification

    def predict(self, data):
        # predict target variables
        # returns and array, where each element is an array of #output scores.
        scores = []
        for pattern in data:
            self.forward(pattern)
            scores.append(self.getOutput())
        return scores

    def dump_weights(self, file_output=None):
        # dump neural network weights to file, calls dump_weights() on each layer
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
    if len(architecture) != len(neurons):
        raise Exception("Architecture miss match")
    if not neurons[0].__name__ is InputNeuron.__name__:
        raise Exception("Input neurons have incorrect type")
    #if not neurons[-1].__name__ is OutputNeuron.__name__:
    #    raise Exception("Output neurons have incorrect type")
    for i in range(1, len(neurons) - 1):
        if neurons[i].__name__ is InputNeuron.__name__ or neurons[i].__name__ is OutputNeuron.__name__:
            raise Exception("Hidden neurons have incorrect type")