from Layer import *


class Network:
    def __init__(self, architecture, neurons, t):
        check_topology(architecture, neurons)
        self.layers = []
        self.target = t
        self.output = [0] * architecture[-1]

        # input layer
        inputNeuron = neurons[0]()
        layer = Layer(architecture[0], 0, inputNeuron)
        self.layers.append(layer)
        # hidden and output layers
        for i in range(1, len(architecture)):
            len_weights = len(self.layers[i-1].neurons)
            neuron = neurons[i](len_weights=len_weights)
            layer = Layer(architecture[i], architecture[i - 1], neuron)
            self.layers.append(layer)

    def forward(self, data):
        self.feed_input_neurons(data)
        self.propagate_input()
        self.set_output()

    def set_output(self):
        # setting output
        last_layer = self.layers[-1]
        for i in range(len(last_layer.neurons) - 1):  # exclude bias
            self.output[i] = last_layer.neurons[i].getOutput()

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

    def back_propagation(self, target, eta=0.1, momentum=0.9):
        # 1. get the output vector from the forward step
        output_net = np.array(self.output)

        # propagate the errors backward through the network:

        # 2. for each network output unit compute its error term delta
        delta_output = self.compute_delta_output_units(output_net, target)

        # 3. for each hidden unit compute its error term delta
        delta_vectors = self.compute_delta_hidden_units(delta_output)

        # 4. update network weights
        # array 3d che contiene i cambiamenti da apportare ai pesi, in particolare delta_w[i][j][k] contiene
        # i cambiamenti da apportare nel layer i+1 (no modifiche ad input layer), neurone j, peso k
        delta_w = self.compute_weight_update(delta_vectors, eta)
            
        return delta_w

    def compute_weight_update(self, delta_vectors, eta):
        delta_w = []
        for i in range(1, len(self.layers)):
            tmpL = []
            for j in range(len(self.layers[i].neurons) - 1):
                tmpN = []
                for w in range(len(self.layers[i].neurons[j].weights)):
                    # qui errore precedente, ad ogni passo il neuronre di cui si prendere l'output
                    # e diverso, tuo codice aveta ...neurons[j] , adesso ...neuron[w].
                    tmpN.append(eta * self.layers[i - 1].neurons[w].output * delta_vectors[-i][j])
                tmpL.append(tmpN)
            delta_w.append(tmpL)
        return delta_w

    def compute_delta_hidden_units(self, delta_output):
        delta_vectors = [delta_output]
        for hidden_layer_index in range(len(self.layers) - 2, 0, -1):
            delta_layer = []
            for h in range(len(self.layers[hidden_layer_index].neurons)):
                downstream = self.layers[hidden_layer_index + 1].neurons[:-1]
                weights = [neuron.weights[h] for neuron in downstream]
                gradient_flow = np.dot(weights, delta_output)
                d_net = self.layers[hidden_layer_index].neurons[h].activation_function_derivative()
                delta_h = gradient_flow * d_net
                delta_layer.append(delta_h)
            delta_vectors.append(delta_layer)

        return delta_vectors

    def compute_delta_output_units(self, output_net, target):
        output_layer = self.layers[-1]
        af_derivatives = np.array([neuron.activation_function_derivative() for neuron in output_layer.neurons[:-1]])
        diff = np.array(target) - output_net
        delta_output = np.multiply(af_derivatives, diff)
        return delta_output

    def BackProp(self,eta):
        # using ,as much as possible, the nomenclature used in back propagation lecture
        delta = []       # vectorn in which i save the output neurons' delta (used after)
        outputLayer = len (self.layers) - 1         #output layer
        deltaW = []  # vector in which save what i'll subtract to last layer (initialized = current last layer)

        for i in self.layers[outputLayer].neurons:
            if not isinstance(i, BiasNeuron):
                deltaW.append(copy.deepcopy(i.weights))

        for i in range (len (self.layers[outputLayer].neurons) -1):         #-1 due to exclude bias
            for j in range (len (self.layers[outputLayer].neurons[i].weights)):
                # oi = partial(current's neuron net)/partial current weight
                oi = self.layers[outputLayer-1].neurons[j].output
                # DF = partial(Error)/partial(input to neuron)
                Df = self.output[i] - self.target[i]
                #Dneuron = partial (output's neuron)/partial (current neuron' snet)/ 
                Dneuron = self.layers[outputLayer].neurons[i].activation_function_derivative()
                if j == 0:
                    delta.append(Df*Dneuron)
                    tmp = Df*Dneuron
                else:
                    tmp = Df* Dneuron
                deltaW[i][j] = oi*tmp
        #hidden layers
        for i in range ( outputLayer -1 ,0,-1) :
            for j in range (len(self.layers[i].neurons) -1) : # .1 due to exclude bias
                for k in range (len(self.layers[i].neurons[j].weights)):
                    sum = 0.0   #sum up to k=output layer's number
                    for s in range ( len (self.layers[outputLayer].neurons ) - 1):
                        # tmp = parial(Error)/partial(neruon's net)
                        tmp = delta[s]
                        #Dnet = partial (net)/partial(o)
                        Dnet = self.layers[outputLayer].neurons[s].weights[j]
                        sum += tmp*Dnet
                    # Dout = partial(current neuron's out)/partial (current neuron's net)
                    Dout = self.layers[i].neurons[j].activation_function_derivative()
                    # Dnet = partial (currents neuron's net)/partial(current analyzerd weight's)
                    Dnet = self.layers[i-1].neurons[k].output
                    # update hiddens neuron's weigths
                    self.layers[i].neurons[j].weights[k] -= eta*sum*Dout*Dnet
        # update output neuron's weigths
        for i in range (len(deltaW)):
            for j in range(len(deltaW[i])):
                self.layers[outputLayer].neurons[i].weights[j] -=eta*deltaW[i][j]

    def update_weights(self, delta_w):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons) - 1):
                for k in range(len(self.layers[i].neurons[j].weights)):
                    self.layers[i].neurons[j].weights[k] += delta_w[i - 1][j][k]

    def train(self):
        # fit the data
        # for each iteration/epoch
        #   self.forward()
        #   self.backprop()
        pass

    def predict(self, data):
        # predict target variable
        # scores = forward(data)
        pass

    # TODO add output_file as a parameter
    def dump_weights(self):
        # dump neural network weights to file
        # calls serialize on each layer
        for layer in self.layers[1:]:  # exclude input layer
            layer.dump_weights()




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